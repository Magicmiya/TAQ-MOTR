import os

from typing import Optional
from collections import defaultdict
from torch.utils.data import Dataset

from .transforms_V2 import MultiComposeV2, transformsV2


class MOTDataset(Dataset):
    mode_list = ["train", "val", "test"]

    def __init__(self, cfg: dict, transform: Optional[object]):
        self.name = cfg['dataset_name']
        self.train_sub_sets = self._normalize_train_sub_sets(cfg.get("train_sub_sets", []))
        dataset_dir = cfg.get("dataset_dir")
        dataset_subdir = cfg.get("dataset_subdir", self.name)
        self.dataset_dir: str = str(dataset_dir if dataset_dir else os.path.join(cfg["dataset_root"], dataset_subdir))
        self.video_name_prefix = cfg.get("video_name_prefix", self.name)
        self.mode = "not set"
        self._sub_dir = {}
        self._gts = {}
        self._vid_idx = {}
        self._idx_vid = {}
        self.eval_videos = self._normalize_video_filter(cfg.get("eval_videos"))
        self.selected_videos = []

        self.batch_size = int(self._first_or_self(cfg.get("batch_size", 1), default=1))
        self.sample_length = int(self._first_or_self(cfg.get("sample_length", cfg.get("sample_lengths", [2])), default=2))
        self.sample_mode = str(self._first_or_self(cfg.get("sample_mode", cfg.get("sample_modes", ["fixed_interval"])), default="fixed_interval"))
        self.sample_interval = int(self._first_or_self(cfg.get("sample_interval", cfg.get("sample_intervals", [1])), default=1))

        self.DE_random_IoU_crop_rate = float(cfg.get("DE_random_IoU_crop_rate", 1.0))
        self.stage_cfg = {}

        self.sample_begin_frames = None
        self.sample_vid_max_frame = None

        # set used data to mode
        self.split_dir = ""
        self.gts = defaultdict(lambda: defaultdict(list))
        self.vid_idx = dict()
        self.idx_vid = dict()
        self._len = 0

        # load data
        for mode in self.mode_list:
            self._load_mot_like_data(mode)

        self.epoch = 0
        self.random_idx = 7
        self.transform: MultiComposeV2 | None = None

    def __getitem__(self, item):
        raise NotImplementedError("Subclasses must implement __getitem__")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_stage(self, stage_cfg: Optional[dict] = None):
        stage_cfg = stage_cfg or {}
        self.stage_cfg = stage_cfg
        if self.mode != "train":
            self.set_mode(mode="train")
        self._apply_train_stage(stage_cfg)
        self._rebuild_train_sampling()
        return self

    def set_mode(self, mode: str):
        assert mode in self.mode_list, "unknown dataset mode {},mode should in {mode_list}"
        self.mode = mode
        self.split_dir = self._sub_dir[mode]
        self.gts = self._gts[mode]
        self.vid_idx = self._vid_idx[mode]
        self.idx_vid = self._idx_vid[mode]
        self.selected_videos = self.get_selected_videos(mode=mode)
        self.transform = transformsV2(mode=mode)

    def eval(self, mode='val', rank=0, world_size=1):
        # self._len should be update here
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __len__(self):
        return self._len

    def _load_mot_like_data(self, mode: str, start_name=None):
        # gt : <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <considered>, <class>,<visibility>
        # det: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>,<visibility>
        # https://motchallenge.net/data/MOT17/
        _sub_dir = os.path.join(self.dataset_dir, mode)
        if os.path.exists(_sub_dir):
            self._sub_dir[mode] = _sub_dir
            prefix = self.video_name_prefix if start_name is None else start_name
            if prefix in (None, ""):
                videos = [vid for vid in sorted(os.listdir(_sub_dir)) if os.path.isdir(os.path.join(_sub_dir, vid))]
            else:
                prefix = str(prefix).lower()
                videos = [vid for vid in sorted(os.listdir(_sub_dir)) if os.path.isdir(os.path.join(_sub_dir, vid)) and vid.lower().startswith(prefix)]
            self._gts[mode] = defaultdict(lambda: defaultdict(list))
            self._vid_idx[mode] = dict()
            self._idx_vid[mode] = dict()
            for vid in videos:
                gt_path = os.path.join(_sub_dir, vid, "gt", "gt.txt")
                if os.path.exists(gt_path):
                    with open(gt_path, encoding="utf-8") as f:
                        for line in f:
                            _frame, _id, _x, _y, _w, _h, _considered, _class, _vis_b = line.strip().split(",")[:9]
                            _frame, _id, _class = map(int, (_frame, _id, _class))
                            _x, _y, _w, _h, _considered, _vis_b = map(float, (_x, _y, _w, _h, _considered, _vis_b))

                            self._gts[mode][vid][_frame].append([_id, _x, _y, _w, _h, _considered, _class, _vis_b])
                self._vid_idx[mode][vid] = len(self._vid_idx[mode])
                self._idx_vid[mode][self._vid_idx[mode][vid]] = vid

    @staticmethod
    def _first_or_self(value, default):
        if isinstance(value, list):
            return value[0] if value else default
        return value if value is not None else default

    @staticmethod
    def _normalize_train_sub_sets(train_sub_sets) -> list[str]:
        if train_sub_sets in (None, "", []):
            return []
        if isinstance(train_sub_sets, str):
            return [train_sub_sets]
        if isinstance(train_sub_sets, (list, tuple, set)):
            normalized = []
            seen = set()
            for sub_set in train_sub_sets:
                sub_set = str(sub_set).strip()
                if not sub_set or sub_set in seen:
                    continue
                normalized.append(sub_set)
                seen.add(sub_set)
            return normalized
        raise TypeError(f"Unsupported train_sub_sets type: {type(train_sub_sets)}")

    def _apply_train_stage(self, stage_cfg: dict):
        if "sample_length" in stage_cfg:
            self.sample_length = int(stage_cfg["sample_length"])
        if "batch_size" in stage_cfg:
            self.batch_size = int(stage_cfg["batch_size"])
        if "sample_mode" in stage_cfg:
            self.sample_mode = str(stage_cfg["sample_mode"])
        if "sample_interval" in stage_cfg:
            self.sample_interval = int(stage_cfg["sample_interval"])
        if "DE_random_IoU_crop_rate" in stage_cfg:
            self.DE_random_IoU_crop_rate = float(stage_cfg["DE_random_IoU_crop_rate"])

    def _rebuild_train_sampling(self):
        self.sample_begin_frames = []
        self.sample_vid_max_frame = {}
        for vid in self.vid_idx.keys():
            frame_ids = sorted(self.gts[vid].keys())
            if not frame_ids:
                continue
            t_min = frame_ids[0]
            t_max = frame_ids[-1]
            self.sample_vid_max_frame[vid] = t_max
            for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                self.sample_begin_frames.append((vid, t))
        self._len = len(self.sample_begin_frames)

    @staticmethod
    def _normalize_video_filter(video_filter):
        if video_filter in (None, "", []):
            return None
        if isinstance(video_filter, str):
            return [video_filter]
        if isinstance(video_filter, (list, tuple, set)):
            normalized = [str(video).strip() for video in video_filter if str(video).strip()]
            return normalized or None
        raise TypeError(f"Unsupported eval_videos type: {type(video_filter)}")

    def get_selected_videos(self, mode: str | None = None) -> list[str]:
        target_mode = self.mode if mode is None else mode
        available = self._vid_idx.get(target_mode, {})
        if not self.eval_videos:
            return list(available.keys())
        missing = [video for video in self.eval_videos if video not in available]
        if missing:
            raise KeyError(f"Videos not found in split '{target_mode}': {missing}")
        return list(self.eval_videos)
