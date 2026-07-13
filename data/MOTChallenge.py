import configparser
import os
from collections import defaultdict
from math import floor
from random import randint

import torch
from PIL import Image

from .baseDataset import MOTDataset


class MOTChallengeDataset(MOTDataset):
    supported_benchmarks = {"MOT17", "MOT20"}
    benchmark_options = {
        "MOT17": {
            "use_detector_suffix": True,
            "default_detector": "FRCNN",
            "detector_variants": ("DPM", "FRCNN", "SDP"),
            "gt_files": {
                "train": ("gt.txt",),
                "train_half": ("gt_train_half.txt",),
                "val_half": ("gt_val_half.txt",),
            },
            "preprocess_json": ("train.json", "train_half.json", "val_half.json", "test.json"),
            "preprocess_hint": "python data/tools/gen_motchallenge_gts.py -P <path to your datasets root> -B MOT17",
        },
        "MOT20": {
            "use_detector_suffix": False,
            "default_detector": None,
            "detector_variants": (),
            "gt_files": {
                "train": ("gt.txt",),
                "train_half": ("gt_half-train.txt", "gt_train_half.txt"),
                "val_half": ("gt_half-val.txt", "gt_val_half.txt"),
            },
            "preprocess_json": ("train.json", "train_half.json", "val_half.json", "test.json"),
            "preprocess_hint": "python data/tools/gen_motchallenge_gts.py -P <path to your datasets root> -B MOT20",
        },
    }
    train_subset_aliases = {
        "train": "train",
        "half_train": "train_half",
        "train_half": "train_half",
        "half_val": "val_half",
        "val": "val_half",
        "val_half": "val_half",
    }
    canonical_train_subsets = ("train", "train_half", "val_half")

    def __init__(self, config: dict, mode: str, transform=None):
        if config["dataset_name"] not in self.supported_benchmarks:
            raise ValueError(f"Unsupported MOTChallenge benchmark: {config['dataset_name']}")

        self.benchmark_cfg = self.benchmark_options[config["dataset_name"]]
        self.use_detector_suffix = bool(self.benchmark_cfg["use_detector_suffix"])
        self.detector_variants = tuple(self.benchmark_cfg["detector_variants"])
        default_detector = self.benchmark_cfg["default_detector"]
        self.primary_detector = None
        if self.use_detector_suffix:
            detector = str(config.get("detector", default_detector)).upper().strip()
            if detector not in self.detector_variants:
                raise ValueError(f"Unsupported detector suffix: {detector}")
            self.primary_detector = detector

        self.runtime_splits: dict[str, str] = {}
        self.frame_offsets: dict[str, dict[str, int]] = defaultdict(dict)
        self.local_seq_lengths: dict[str, dict[str, int]] = defaultdict(dict)
        self.full_seq_lengths: dict[str, int] = {}
        self.seq_img_ext: dict[str, str] = {}
        self.runtime_split_views: dict[str, dict] = {}
        super().__init__(cfg=config, transform=transform)
        self._validate_preprocessed_artifacts(runtime_mode=mode)
        self.set_mode(mode=mode)
        if mode == "train":
            self.set_epoch(0)
            self.set_stage()

    def __getitem__(self, sampler_info):
        idx, item = sampler_info
        self.random_idx = idx
        sample_info = self.sample_begin_frames[item]
        if len(sample_info) == 3:
            subset_name, vid, begin_frame = sample_info
            vid_key = (subset_name, vid)
        else:
            subset_name = None
            vid, begin_frame = sample_info
            vid_key = vid
        frame_idxs = self._sample_frames_idx(vid=vid_key, begin_frame=begin_frame)
        imgs, infos = self._get_multi_frames(vid=vid, idxs=frame_idxs, subset_name=subset_name)
        if self.transform is not None:
            imgs, infos = self.transform(imgs, infos, item, self)
        return {"imgs": imgs, "infos": infos}

    def __len__(self):
        assert self.sample_begin_frames is not None, "Please use set_stage to init MOTChallenge dataset."
        return self._len

    def set_mode(self, mode: str):
        super().set_mode(mode=mode)
        self.active_split = self.runtime_splits[mode]

    def eval(self, mode="val", rank=0, world_size=1):
        self.set_mode(mode)
        self.sample_begin_frames = []
        self.sample_vid_max_frame = {}
        self.sample_mode = "fixed_interval"
        self.sample_length = 1
        self.sample_interval = 1
        for vid in self.selected_videos:
            if self.vid_idx[vid] % world_size != rank:
                continue
            frame_ids = self._get_eval_frame_ids(vid)
            if not frame_ids:
                continue
            self.sample_vid_max_frame[vid] = frame_ids[-1]
            self.sample_begin_frames += [(vid, t) for t in frame_ids]
        self.batch_size = 1
        self._len = len(self.sample_begin_frames)
        return self

    def _load_mot_like_data(self, mode: str, start_name=None):
        runtime_split = self._resolve_runtime_split(mode)
        view = self._load_runtime_split_view(runtime_split=runtime_split)
        if view is None:
            return
        self.runtime_splits[mode] = runtime_split
        self._sub_dir[mode] = view["split_dir"]
        self._gts[mode] = view["gts"]
        self._vid_idx[mode] = view["vid_idx"]
        self._idx_vid[mode] = view["idx_vid"]
        self.frame_offsets[mode] = dict(view["frame_offsets"])
        self.local_seq_lengths[mode] = dict(view["local_seq_lengths"])

    def _load_runtime_split_view(self, runtime_split: str) -> dict | None:
        if runtime_split in self.runtime_split_views:
            return self.runtime_split_views[runtime_split]

        storage_split = self._resolve_storage_split_from_runtime(runtime_split)
        split_dir = os.path.join(self.dataset_dir, storage_split)
        if not os.path.isdir(split_dir):
            return None

        view = {
            "runtime_split": runtime_split,
            "storage_split": storage_split,
            "split_dir": split_dir,
            "gts": defaultdict(lambda: defaultdict(list)),
            "vid_idx": {},
            "idx_vid": {},
            "frame_offsets": {},
            "local_seq_lengths": {},
        }

        for vid in self._list_videos(split_dir):
            view["vid_idx"][vid] = len(view["vid_idx"])
            view["idx_vid"][view["vid_idx"][vid]] = vid

            full_seq_length = self._read_seq_length(storage_split, vid)
            self.full_seq_lengths[vid] = full_seq_length
            self.seq_img_ext[vid] = self._read_image_extension(storage_split, vid)
            view["frame_offsets"][vid] = self._resolve_frame_offset(runtime_split, full_seq_length)
            view["local_seq_lengths"][vid] = self._resolve_local_seq_length(runtime_split, full_seq_length)

            gt_path = self._resolve_gt_path(split_dir=split_dir, vid=vid, runtime_split=runtime_split)
            if gt_path is None:
                continue
            self._load_gt_file(gts=view["gts"], vid=vid, gt_path=gt_path)

        self.runtime_split_views[runtime_split] = view
        return view

    def _load_gt_file(self, gts: dict, vid: str, gt_path: str):
        with open(gt_path, encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split(",")
                if len(fields) < 9:
                    continue
                frame_idx, track_id, cls = map(int, (fields[0], fields[1], fields[7]))
                x, y, w, h, considered, vis = map(float, (fields[2], fields[3], fields[4], fields[5], fields[6], fields[8]))
                if not self._keep_training_annotation(considered=considered, cls=cls):
                    continue
                gts[vid][frame_idx].append([track_id, x, y, w, h, considered, cls, vis])

    def _resolve_runtime_split(self, mode: str) -> str:
        if mode == "train":
            return "train"
        if mode == "val":
            return "val_half"
        if mode == "test":
            return "test"
        raise ValueError(f"Unsupported runtime mode: {mode}")

    @staticmethod
    def _resolve_storage_split(mode: str) -> str:
        return "train" if mode in {"train", "val"} else "test"

    @staticmethod
    def _resolve_storage_split_from_runtime(runtime_split: str) -> str:
        return "test" if runtime_split == "test" else "train"

    def _resolve_gt_file_candidates(self, runtime_split: str) -> tuple[str, ...] | None:
        if runtime_split == "test":
            return None
        candidates = self.benchmark_cfg["gt_files"].get(runtime_split)
        if not candidates:
            raise ValueError(f"Unsupported runtime split: {runtime_split}")
        return tuple(candidates)

    def _resolve_gt_path(self, split_dir: str, vid: str, runtime_split: str) -> str | None:
        candidates = self._resolve_gt_file_candidates(runtime_split)
        if candidates is None:
            return None
        for file_name in candidates:
            gt_path = os.path.join(split_dir, vid, "gt", file_name)
            if os.path.isfile(gt_path):
                return gt_path
        gt_dir = os.path.join(split_dir, vid, "gt")
        raise FileNotFoundError(
            f"Missing GT file for {self.name} {runtime_split} split: {vid}. "
            f"Tried {list(candidates)} under {gt_dir}. "
            f"Please run `{self.benchmark_cfg['preprocess_hint']}` first."
        )

    def _list_videos(self, split_dir: str) -> list[str]:
        videos = [video for video in sorted(os.listdir(split_dir)) if os.path.isdir(os.path.join(split_dir, video)) and video.startswith(self.name)]
        if not self.use_detector_suffix:
            return videos
        detector_suffix = f"-{self.primary_detector}"
        return [video for video in videos if video.endswith(detector_suffix)]

    def _read_seq_length(self, storage_split: str, vid: str) -> int:
        seqinfo_path = os.path.join(self.dataset_dir, storage_split, vid, "seqinfo.ini")
        parser = configparser.ConfigParser()
        parser.read(seqinfo_path)
        return int(parser["Sequence"]["seqLength"])

    def _read_image_extension(self, storage_split: str, vid: str) -> str:
        seqinfo_path = os.path.join(self.dataset_dir, storage_split, vid, "seqinfo.ini")
        parser = configparser.ConfigParser()
        parser.read(seqinfo_path)
        return parser["Sequence"].get("imExt", ".jpg")

    @staticmethod
    def _resolve_frame_offset(runtime_split: str, full_seq_length: int) -> int:
        if runtime_split == "val_half":
            return full_seq_length // 2
        return 0

    @staticmethod
    def _resolve_local_seq_length(runtime_split: str, full_seq_length: int) -> int:
        if runtime_split == "train_half":
            return full_seq_length // 2
        if runtime_split == "val_half":
            return full_seq_length - (full_seq_length // 2)
        return full_seq_length

    @staticmethod
    def _keep_training_annotation(considered: float, cls: int) -> bool:
        return considered > 0 and cls == 1

    def _validate_preprocessed_artifacts(self, runtime_mode: str):
        if runtime_mode not in {"train", "val"}:
            return
        annotations_dir = os.path.join(self.dataset_dir, "annotations")
        missing_annotations = [
            file_name
            for file_name in self.benchmark_cfg["preprocess_json"]
            if not os.path.isfile(os.path.join(annotations_dir, file_name))
        ]
        if missing_annotations:
            raise FileNotFoundError(
                f"Missing preprocessed annotation files for {self.name}: {missing_annotations}. "
                f"Please run `{self.benchmark_cfg['preprocess_hint']}` first."
            )

    def _sample_frames_idx(self, vid: str, begin_frame: int) -> list[int]:
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample length is less than 2 at random_interval."
            remain_frames = self.sample_vid_max_frame[vid] - begin_frame
            max_interval = floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            return [begin_frame + interval * i for i in range(self.sample_length)]
        if self.sample_mode == "fixed_interval":
            assert self.sample_length > 0, "Sample length is less than 1 at fixed_interval."
            remain_frames = self.sample_vid_max_frame[vid] - begin_frame
            max_interval = remain_frames if self.sample_length == 1 else floor(remain_frames / (self.sample_length - 1))
            interval = min(self.sample_interval, max_interval)
            return [begin_frame + interval * i for i in range(self.sample_length)]
        raise ValueError(f"Sample mode {self.sample_mode} is not supported.")

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

        if self.mode == "val":
            offset = self.frame_offsets[self.mode].get(vid, 0)
            return [frame_id - offset for frame_id in frame_ids if frame_id > offset]
        return frame_ids

    def _get_runtime_view_for_subset(self, subset_name: str) -> dict:
        view = self._load_runtime_split_view(runtime_split=subset_name)
        if view is None:
            raise FileNotFoundError(f"{self.name} runtime split '{subset_name}' is unavailable.")
        return view

    def _get_default_train_subset_name(self) -> str:
        return "train"

    def _get_train_subset_names(self) -> list[str]:
        if not self.train_sub_sets:
            return [self._get_default_train_subset_name()]
        normalized = []
        seen = set()
        for subset_name in self.train_sub_sets:
            subset_key = str(subset_name).strip()
            if subset_key not in self.train_subset_aliases:
                raise KeyError(f"{self.name} train_sub_sets contains unsupported subset '{subset_name}'")
            subset_key = self.train_subset_aliases[subset_key]
            if subset_key in seen:
                continue
            normalized.append(subset_key)
            seen.add(subset_key)
        return normalized

    def _get_image_frame_idx(self, vid: str, local_frame_idx: int, subset_name: str | None = None) -> int:
        if subset_name is None:
            return local_frame_idx + self.frame_offsets[self.mode].get(vid, 0)
        return local_frame_idx + self._get_runtime_view_for_subset(subset_name)["frame_offsets"].get(vid, 0)

    def _get_single_frame(self, vid: str, idx: int, subset_name: str | None = None):
        if subset_name is None:
            split_dir = self.split_dir
            vid_idx = self.vid_idx
            gts = self.gts
            subset_token = self.mode
        else:
            view = self._get_runtime_view_for_subset(subset_name)
            split_dir = view["split_dir"]
            vid_idx = view["vid_idx"]
            gts = view["gts"]
            subset_token = subset_name

        image_idx = self._get_image_frame_idx(vid=vid, local_frame_idx=idx, subset_name=subset_name)
        image_ext = self.seq_img_ext.get(vid, ".jpg")
        img_path = os.path.join(split_dir, vid, "img1", f"{image_idx:06d}{image_ext}")
        img = Image.open(img_path)
        subset_offset = self.canonical_train_subsets.index(subset_token) * 10000000 if subset_name is not None else 0
        ids_offset = subset_offset + vid_idx[vid] * 100000

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
            "source_subset": subset_token,
            "reset_memory_each_frame": False,
        }
        for track_id, x, y, w, h, considered, cls, vis in gts[vid][idx]:
            info["bbox"].append([float(x), float(y), float(w), float(h)])
            info["areas"].append(w * h)
            info["ids"].append(track_id + ids_offset)
            info["labels"].append(0)
        info["bbox"] = torch.as_tensor(info["bbox"])
        info["areas"] = torch.as_tensor(info["areas"])
        info["ids"] = torch.as_tensor(info["ids"])
        info["labels"] = torch.as_tensor(info["labels"])
        if len(info["bbox"]) > 0:
            info["bbox"][:, 2:] += info["bbox"][:, :2]
        else:
            info["bbox"] = torch.zeros((0, 4))
            info["ids"] = torch.zeros((0,), dtype=torch.long)
            info["labels"] = torch.zeros((0,), dtype=torch.long)
        return img, info

    def _get_multi_frames(self, vid: str, idxs: list[int], subset_name: str | None = None):
        return zip(*[self._get_single_frame(vid=vid, idx=idx, subset_name=subset_name) for idx in idxs])

    def _rebuild_train_sampling(self):
        self.sample_begin_frames = []
        self.sample_vid_max_frame = {}
        for subset_name in self._get_train_subset_names():
            view = self._get_runtime_view_for_subset(subset_name=subset_name)
            for vid in view["vid_idx"].keys():
                frame_ids = sorted(view["gts"][vid].keys())
                if not frame_ids:
                    continue
                t_min = frame_ids[0]
                t_max = frame_ids[-1]
                vid_key = (subset_name, vid)
                self.sample_vid_max_frame[vid_key] = t_max
                for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                    self.sample_begin_frames.append((subset_name, vid, t))
        self._len = len(self.sample_begin_frames)

    def get_selected_videos(self, mode: str | None = None) -> list[str]:
        target_mode = self.mode if mode is None else mode
        available = self._vid_idx.get(target_mode, {})
        if not self.eval_videos:
            return list(available.keys())

        selected = []
        missing = []
        for video in self.eval_videos:
            normalized = video
            if self.use_detector_suffix and video not in available:
                normalized = f"{video}-{self.primary_detector}"
            if normalized in available:
                selected.append(normalized)
            else:
                missing.append(video)
        if missing:
            raise KeyError(f"Videos not found in split '{target_mode}': {missing}")
        return selected

    def get_trackeval_dataset_config(self, mode: str | None = None) -> dict:
        target_mode = self.mode if mode is None else mode
        runtime_split = self.runtime_splits[target_mode]
        storage_split = self._resolve_storage_split(target_mode)
        gt_candidates = self._resolve_gt_file_candidates(runtime_split)
        if gt_candidates is None:
            raise ValueError(f"Runtime split '{runtime_split}' does not expose GT files.")
        gt_file_name = gt_candidates[0]
        return {
            "GT_FOLDER": os.path.join(self.dataset_dir, storage_split),
            "SPLIT_TO_EVAL": storage_split,
            "SEQ_INFO": {video: self.local_seq_lengths[target_mode][video] for video in self.get_selected_videos(target_mode)},
            "GT_LOC_FORMAT": f"{{gt_folder}}/{{seq}}/gt/{gt_file_name}",
            "BENCHMARK": self.name,
        }

    def get_submission_layout(self) -> dict[str, list[str] | tuple[str, ...] | str]:
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")
        return {
            "primary_detector": self.primary_detector,
            "detectors": self.detector_variants,
            "use_detector_suffix": self.use_detector_suffix,
            "train_videos": self._list_videos(train_dir),
            "test_videos": self._list_videos(test_dir),
        }


def build(config: dict, mode: str):
    return MOTChallengeDataset(config=config, mode=mode)
