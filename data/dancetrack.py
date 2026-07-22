import os
from math import floor
from random import randint

import torch
from PIL import Image
from .baseDataset import MOTDataset


class DanceTrack(MOTDataset):

    def __init__(self, config: dict, mode: str, transform=None):
        super(DanceTrack, self).__init__(cfg=config, transform=transform)
        self.set_mode(mode=mode)
        if mode == 'train':
            self.set_epoch(0)
            self.set_stage()
        return

    def __getitem__(self, sampler_info):
        idx, item = sampler_info
        self.random_idx = idx
        sample_info = self.sample_begin_frames[item]
        if len(sample_info) == 3:
            subset_name, vid, begin_frame = sample_info
            vid_key = (subset_name, vid)
        else:
            subset_name = self.mode
            vid, begin_frame = sample_info
            vid_key = vid
        frame_idxs = self._sample_frames_idx(vid=vid_key, begin_frame=begin_frame)
        imgs, infos = self._get_multi_frames(vid=vid, idxs=frame_idxs, subset_name=subset_name)
        if self.transform is not None:
            imgs, infos = self.transform(imgs, infos, item, self)
        return {"imgs": imgs, "infos": infos}

    def __len__(self):
        assert self.sample_begin_frames is not None, "Please use set_stage to init DanceTrack Dataset."
        return self._len

    def eval(self, mode='val', rank=0, world_size=1):
        self.set_mode(mode)
        self.sample_begin_frames = list()
        self.sample_vid_max_frame = dict()
        self.sample_mode = "fixed_interval"
        self.sample_length = 1
        self.sample_interval = 1
        for vid in self.selected_videos:
            if self.vid_idx[vid] % world_size == rank:
                frame_ids = self._get_eval_frame_ids(vid)
                t_min = frame_ids[0]
                t_max = frame_ids[-1]
                self.sample_vid_max_frame[vid] = t_max
                self.sample_begin_frames += [(vid, t) for t in frame_ids]
            else:
                continue
        self.batch_size = 1
        self._len = len(self.sample_begin_frames)
        return self

    def _get_eval_frame_ids(self, vid: str) -> list[int]:
        gt_frame_ids = sorted(self.gts[vid].keys())
        if gt_frame_ids:
            return gt_frame_ids

        img_dir = os.path.join(self.split_dir, vid, "img1")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found for video {vid}: {img_dir}")

        frame_ids = []
        for file_name in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(file_name)
            if ext.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            if stem.isdigit():
                frame_ids.append(int(stem))

        if not frame_ids:
            raise FileNotFoundError(f"No image frames found for video {vid}: {img_dir}")
        return frame_ids

    def _sample_frames_idx(self, vid: str | tuple[str, str], begin_frame: int) -> list[int]:
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample length is less than 2 at random_interval."
            remain_frames = self.sample_vid_max_frame[vid] - begin_frame
            max_interval = floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            frame_idxs = [
                begin_frame + interval * i for i in range(self.sample_length)
            ]
            return frame_idxs
        elif self.sample_mode == "fixed_interval":
            assert self.sample_length > 0, "Sample length is less than 1 at fixed_interval."
            remain_frames = self.sample_vid_max_frame[vid] - begin_frame
            max_interval = remain_frames if self.sample_length == 1 else floor(
                remain_frames / (self.sample_length - 1))
            interval = min(self.sample_interval, max_interval)
            frame_idxs = [
                begin_frame + interval * i for i in range(self.sample_length)
            ]
            return frame_idxs

        else:
            raise ValueError(
                f"Sample mode {self.sample_mode} is not supported.")

    def _get_single_frame(self, vid: str, idx: int):
        return self._get_single_frame_from_subset(vid=vid, idx=idx, subset_name=self.mode)

    def _get_single_frame_from_subset(self, vid: str, idx: int, subset_name: str):
        split_dir = self._sub_dir[subset_name]
        img_path = os.path.join(
            split_dir, vid, "img1", f"{idx:0{self.image_name_width}d}.jpg")
        img = Image.open(img_path)
        subset_offset = self.mode_list.index(subset_name) * 10000000
        ids_offset = subset_offset + self._vid_idx[subset_name][vid] * 100000

        # label GT：
        info = {
            "bbox": [],
            "ids": [],
            "labels": [],
            "areas": [],
            "frame_idx": torch.as_tensor(idx),
            "video_name": vid,
            "org_shape": img.size,
            "img_path": img_path,
            "source_dataset": self.name,
            "source_subset": subset_name,
            "reset_memory_each_frame": False,
        }
        for _id, _x, _y, _w, _h, _considered, _class, _vis_b in self._gts[subset_name][vid][
                idx]:
            info["bbox"].append(list(map(float, (_x, _y, _w, _h))))
            info["areas"].append(_w * _h)  # area = w * h
            info["ids"].append(_id + ids_offset)
            info["labels"].append(0)  # DanceTrack, all people.
        info["bbox"] = torch.as_tensor(info["bbox"])
        info["areas"] = torch.as_tensor(info["areas"])
        info["ids"] = torch.as_tensor(info["ids"])
        info["labels"] = torch.as_tensor(info["labels"])
        # xywh to x1y1x2y2
        if len(info["bbox"]) > 0:
            info["bbox"][:, 2:] += info["bbox"][:, :2]
        else:
            info["bbox"] = torch.zeros((0, 4))
            info["ids"] = torch.zeros((0, ), dtype=torch.long)
            info["labels"] = torch.zeros((0, ), dtype=torch.long)

        return img, info

    def _get_multi_frames(self, vid: str, idxs: list[int], subset_name: str | None = None):
        subset_name = self.mode if subset_name is None else subset_name
        return zip(*[self._get_single_frame_from_subset(vid=vid, idx=i, subset_name=subset_name) for i in idxs])

    def _get_train_subset_names(self) -> list[str]:
        if self.train_sub_sets:
            return self.train_sub_sets
        return ["train"]

    def _rebuild_train_sampling(self):
        self.sample_begin_frames = []
        self.sample_vid_max_frame = {}
        for subset_name in self._get_train_subset_names():
            if subset_name not in self._gts:
                raise KeyError(f"{self.name} train_sub_sets contains unsupported subset '{subset_name}'")
            subset_gts = self._gts[subset_name]
            subset_vid_idx = self._vid_idx[subset_name]
            for vid in subset_vid_idx.keys():
                frame_ids = sorted(subset_gts[vid].keys())
                if not frame_ids:
                    continue
                t_min = frame_ids[0]
                t_max = frame_ids[-1]
                vid_key = (subset_name, vid)
                self.sample_vid_max_frame[vid_key] = t_max
                for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                    self.sample_begin_frames.append((subset_name, vid, t))
        self._len = len(self.sample_begin_frames)


def build(config: dict, mode: str):
    return DanceTrack(config=config, mode=mode)
