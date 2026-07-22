from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image

from .aux_det import load_coco_aux_det_source
from .baseDataset import MOTDataset
from .transforms_V2 import transformsV2


def _load_crowdhuman_records(annotation_path: str) -> list[dict]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def _resolve_crowdhuman_image(dataset_root: str, split: str, image_id: str) -> str:
    split_dirs = sorted(Path(dataset_root).glob(f"CrowdHuman_{split}*/Images"))
    for split_dir in split_dirs:
        img_path = split_dir / f"{image_id}.jpg"
        if img_path.is_file():
            return str(img_path)
    raise FileNotFoundError(f"CrowdHuman image not found for split={split}, image_id={image_id}")


def _load_crowdhuman_items(dataset_root: str, split: str) -> list[dict]:
    annotation_path = Path(dataset_root) / f"annotation_{split}.odgt"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"CrowdHuman annotation file not found: {annotation_path}")

    items = []
    for record in _load_crowdhuman_records(str(annotation_path)):
        annotations = []
        for ann in record.get("gtboxes", []):
            extra = ann.get("extra", {})
            if int(extra.get("ignore", 0)) != 0:
                continue
            x, y, w, h = map(float, ann["fbox"])
            if w <= 0 or h <= 0:
                continue
            annotations.append({"bbox": [x, y, w, h]})
        items.append(
            {
                "image_id": str(record["ID"]),
                "img_path": _resolve_crowdhuman_image(dataset_root=dataset_root, split=split, image_id=str(record["ID"])),
                "annotations": annotations,
            }
        )
    if not items:
        raise RuntimeError(f"CrowdHuman split '{split}' has no valid items.")
    return items


class ImageDataset(MOTDataset):
    supported_datasets = {"CrowdHuman", "CityPersons"}

    def __init__(self, config: dict, mode: str, transform=None):
        dataset_name = str(config["dataset_name"]).strip()
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Unsupported image dataset: {dataset_name}")

        self.items_by_mode: dict[str, list[dict]] = {}
        self.mode_item_indices: list[tuple[str, int]] = []
        self.missing_modes: set[str] = set()
        pseudo_motion_cfg = config.get("image_pseudo_motion", {})
        if not isinstance(pseudo_motion_cfg, dict):
            pseudo_motion_cfg = {}
        self.pseudo_motion_shift = bool(pseudo_motion_cfg.get("enabled", False))
        self.pseudo_motion_shift_max_ratio = float(pseudo_motion_cfg.get("max_shift_ratio", 0.06))
        self.pseudo_motion_shift_reverse_prob = float(pseudo_motion_cfg.get("reverse_prob", 0.5))
        self.pseudo_motion_shift_overflow_bbox = bool(pseudo_motion_cfg.get("overflow_bbox", False))
        super(ImageDataset, self).__init__(cfg=config, transform=transform)
        self.set_mode(mode=mode)
        if mode == "train":
            self.set_epoch(0)
            self.set_stage()

    def _load_mot_like_data(self, mode: str, start_name=None):
        if mode == "test":
            self._sub_dir[mode] = self.dataset_dir
            self._gts[mode] = defaultdict(lambda: defaultdict(list))
            self._vid_idx[mode] = {}
            self._idx_vid[mode] = {}
            self.items_by_mode[mode] = []
            return

        self._sub_dir[mode] = self.dataset_dir
        self._gts[mode] = defaultdict(lambda: defaultdict(list))
        self._vid_idx[mode] = {f"{self.name}_{mode}": 0}
        self._idx_vid[mode] = {0: f"{self.name}_{mode}"}
        try:
            self.items_by_mode[mode] = self._load_items(mode=mode)
        except FileNotFoundError:
            self.items_by_mode[mode] = []
            self.missing_modes.add(mode)

    def _load_items(self, mode: str) -> list[dict]:
        if self.name == "CrowdHuman":
            return _load_crowdhuman_items(dataset_root=self.dataset_dir, split=mode)
        if self.name == "CityPersons":
            annotations_path = str(Path(self.dataset_dir) / "annotations" / f"{mode}.json")
            source = load_coco_aux_det_source(name=self.name, dataset_root=self.dataset_dir, annotations_path=annotations_path)
            return list(source.items)
        raise ValueError(f"Unsupported image dataset: {self.name}")

    def set_mode(self, mode: str):
        if mode != "train":
            raise NotImplementedError(f"{self.name} only supports train mode.")
        self.mode = mode
        self.split_dir = self.dataset_dir
        self.gts = self._gts[mode]
        self.vid_idx = self._vid_idx[mode]
        self.idx_vid = self._idx_vid[mode]
        self.selected_videos = list(self.vid_idx.keys())
        self.transform = self.transform if self.transform is not None else transformsV2(mode=mode)

    def eval(self, mode='val', rank=0, world_size=1):
        raise NotImplementedError(f"{self.name} does not support eval mode.")

    def _get_train_subset_names(self) -> list[str]:
        if self.train_sub_sets:
            return self.train_sub_sets
        return ["train"]

    def _rebuild_train_sampling(self):
        self.mode_item_indices = []
        for subset_name in self._get_train_subset_names():
            if subset_name in self.missing_modes:
                raise FileNotFoundError(f"{self.name} does not provide requested subset '{subset_name}'")
            if subset_name not in self.items_by_mode:
                raise KeyError(f"{self.name} train_sub_sets contains unsupported subset '{subset_name}'")
            self.mode_item_indices.extend((subset_name, idx) for idx in range(len(self.items_by_mode[subset_name])))
        self.sample_begin_frames = list(self.mode_item_indices)
        self._len = len(self.mode_item_indices)

    def __getitem__(self, sampler_info):
        random_idx, item = sampler_info
        self.random_idx = random_idx
        subset_name, local_idx = self.mode_item_indices[item]
        imgs, infos = self._build_repeat_clip(subset_name=subset_name, local_idx=local_idx)
        if self.pseudo_motion_shift:
            imgs, infos = self._apply_pseudo_motion_shift(imgs=imgs, infos=infos)
        if self.transform is not None:
            imgs, infos = self.transform(imgs, infos, item, self)
        return {"imgs": imgs, "infos": infos}

    def _build_repeat_clip(self, subset_name: str, local_idx: int):
        item = self.items_by_mode[subset_name][local_idx]
        img = Image.open(item["img_path"])
        width, height = img.size
        infos = []
        imgs = []
        for frame_idx in range(self.sample_length):
            info = {
                "bbox": [],
                "ids": [],
                "labels": [],
                "areas": [],
                "frame_idx": torch.as_tensor(frame_idx + 1),
                "video_name": f"{self.name}_{subset_name}_{local_idx:06d}",
                "org_shape": (width, height),
                "img_path": item["img_path"],
                "source_dataset": self.name,
                "source_subset": subset_name,
                "reset_memory_each_frame": False,
            }
            for ann_idx, ann in enumerate(item["annotations"]):
                x, y, w, h = map(float, ann["bbox"])
                info["bbox"].append([x, y, w, h])
                info["areas"].append(w * h)
                info["ids"].append(ann_idx)
                info["labels"].append(0)

            info["bbox"] = torch.as_tensor(info["bbox"])
            info["areas"] = torch.as_tensor(info["areas"])
            info["ids"] = torch.as_tensor(info["ids"], dtype=torch.long)
            info["labels"] = torch.as_tensor(info["labels"], dtype=torch.long)
            if len(info["bbox"]) > 0:
                info["bbox"][:, 2:] += info["bbox"][:, :2]
            else:
                info["bbox"] = torch.zeros((0, 4))
                info["ids"] = torch.zeros((0,), dtype=torch.long)
                info["labels"] = torch.zeros((0,), dtype=torch.long)

            imgs.append(img.copy())
            infos.append(info)
        return imgs, infos


    def _apply_pseudo_motion_shift(self, imgs: list[Image.Image], infos: list[dict]):
        if len(imgs) <= 1 or not self._is_pseudo_motion_clip_legal(infos):
            return imgs, infos

        width, height = imgs[0].size
        max_x_shift = math.ceil(self.pseudo_motion_shift_max_ratio * width)
        max_y_shift = math.ceil(self.pseudo_motion_shift_max_ratio * height)
        if max_x_shift <= 0 and max_y_shift <= 0:
            return imgs, infos

        x_shift = random.randint(-max_x_shift, max_x_shift) if max_x_shift > 0 else 0
        y_shift = random.randint(-max_y_shift, max_y_shift) if max_y_shift > 0 else 0
        left = max(0, x_shift)
        right = min(width, width + x_shift)
        top = max(0, y_shift)
        bottom = min(height, height + y_shift)
        crop_w = int(right - left)
        crop_h = int(bottom - top)
        if crop_w <= 0 or crop_h <= 0:
            return imgs, infos

        # Simulate FDTA-style pseudo motion only for repeated image clips before the shared train transforms.
        shifted_imgs = [imgs[0].copy()]
        shifted_infos = [deepcopy(infos[0])]
        for frame_idx in range(1, len(imgs)):
            prev_img = shifted_imgs[-1].copy()
            prev_info = deepcopy(shifted_infos[-1])
            img_i = self._crop_resize_image(
                image=prev_img,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                target_size=(width, height),
            )
            info_i = self._shift_resize_info(
                info=prev_info,
                frame_template=infos[frame_idx],
                left=left,
                top=top,
                crop_w=crop_w,
                crop_h=crop_h,
                target_w=width,
                target_h=height,
            )
            shifted_imgs.append(img_i)
            shifted_infos.append(info_i)

        if not self._is_pseudo_motion_clip_legal(shifted_infos):
            return imgs, infos

        if random.random() < self.pseudo_motion_shift_reverse_prob:
            shifted_imgs.reverse()
            shifted_infos.reverse()
            for frame_idx, info in enumerate(shifted_infos):
                info["frame_idx"] = infos[frame_idx]["frame_idx"]
        return shifted_imgs, shifted_infos

    @staticmethod
    def _crop_resize_image(
        image: Image.Image,
        left: int,
        top: int,
        right: int,
        bottom: int,
        target_size: tuple[int, int],
    ) -> Image.Image:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return image.crop((left, top, right, bottom)).resize(target_size, resample=resampling)

    def _shift_resize_info(
        self,
        info: dict,
        frame_template: dict,
        left: int,
        top: int,
        crop_w: int,
        crop_h: int,
        target_w: int,
        target_h: int,
    ) -> dict:
        boxes = info["bbox"].clone()
        boxes = boxes - torch.as_tensor([left, top, left, top], dtype=boxes.dtype)
        if boxes.numel() > 0:
            clipped_boxes = boxes.clone().reshape(-1, 2, 2)
            max_wh = torch.as_tensor([crop_w, crop_h], dtype=boxes.dtype)
            clipped_boxes = torch.min(clipped_boxes, max_wh).clamp(min=0)
            keep = torch.all(clipped_boxes[:, 1, :] > clipped_boxes[:, 0, :], dim=1)
            if self.pseudo_motion_shift_overflow_bbox:
                boxes = boxes.reshape(-1, 4)
            else:
                boxes = clipped_boxes.reshape(-1, 4)
            boxes = boxes[keep]
            scale = torch.as_tensor(
                [target_w / crop_w, target_h / crop_h, target_w / crop_w, target_h / crop_h],
                dtype=boxes.dtype,
            )
            boxes = boxes * scale
            info["bbox"] = boxes
            info["ids"] = info["ids"][keep]
            info["labels"] = info["labels"][keep]
            wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
            info["areas"] = wh[:, 0] * wh[:, 1]
        else:
            info["bbox"] = torch.zeros((0, 4), dtype=torch.float32)
            info["ids"] = torch.zeros((0,), dtype=torch.long)
            info["labels"] = torch.zeros((0,), dtype=torch.long)
            info["areas"] = torch.zeros((0,), dtype=torch.float32)

        info["frame_idx"] = frame_template["frame_idx"]
        info["org_shape"] = frame_template["org_shape"]
        return info

    @staticmethod
    def _is_pseudo_motion_clip_legal(infos: list[dict]) -> bool:
        for info in infos:
            bbox = info.get("bbox")
            ids = info.get("ids")
            labels = info.get("labels")
            areas = info.get("areas")
            if bbox is None or ids is None or labels is None or areas is None:
                return False
            if bbox.ndim != 2 or bbox.shape[-1] != 4 or bbox.shape[0] == 0:
                return False
            if len(ids) != bbox.shape[0] or len(labels) != bbox.shape[0] or len(areas) != bbox.shape[0]:
                return False
            # Keep conservative fallback behavior when pseudo motion removes all valid targets.
            if torch.unique(ids).numel() != ids.numel():
                return False
        return True


def build(config: dict, mode: str):
    return ImageDataset(config=config, mode=mode)
