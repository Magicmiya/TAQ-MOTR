from __future__ import annotations

from typing import Optional

from .baseDataset import MOTDataset
from .builders import build_named_dataset, normalize_train_dataset_configs


class MixedDataset(MOTDataset):
    def __init__(self, config: dict, mode: str, transform=None):
        if mode != "train":
            raise ValueError("MixedDataset only supports train mode.")

        self.config = dict(config)
        self.name = "MixedDataset"
        self.dataset_dir = ""
        self.mode = "train"
        self.train_sub_sets = []
        self.video_name_prefix = self.name
        self.eval_videos = None
        self.selected_videos = []
        self.sample_begin_frames = []
        self.sample_vid_max_frame = {}
        self.split_dir = ""
        self.gts = {}
        self.vid_idx = {}
        self.idx_vid = {}
        self._len = 0
        self.epoch = 0
        self.random_idx = 7
        self.transform = None
        self.batch_size = int(config.get("batch_size", 1))
        self.sample_length = int(config.get("sample_length", 2))
        self.sample_mode = str(config.get("sample_mode", "fixed_interval"))
        self.sample_interval = int(config.get("sample_interval", 1))
        self.active_datasets = []
        self.global_index_map: list[tuple[str, int]] = []

        self.datasets_by_name = {}
        for dataset_cfg in normalize_train_dataset_configs(config):
            dataset = build_named_dataset(config=dataset_cfg, mode=mode)
            self.datasets_by_name[dataset.name] = dataset
        self.dataset_names = list(self.datasets_by_name.keys())
        if not self.dataset_names:
            raise RuntimeError("MixedDataset requires at least one train dataset.")
        self.eval_dataset_name = self.dataset_names[0]

        self.set_epoch(0)
        self.set_stage()

    def __getitem__(self, sampler_info):
        if self.mode != "train":
            return self._get_eval_dataset()[sampler_info]
        random_idx, item = sampler_info
        dataset_name, local_idx = self.global_index_map[item]
        dataset = self.datasets_by_name[dataset_name]
        return dataset[(random_idx, local_idx)]

    def __len__(self):
        return self._len

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        for dataset in self.datasets_by_name.values():
            dataset.set_epoch(epoch)

    def set_stage(self, stage_cfg: Optional[dict] = None):
        stage_cfg = stage_cfg or {}
        self._restore_train_view()
        self._apply_train_stage(stage_cfg)
        requested_names = self._normalize_active_dataset_names(stage_cfg.get("active_datasets", self.dataset_names))
        unknown_names = [name for name in requested_names if name not in self.datasets_by_name]
        if unknown_names:
            raise KeyError(f"Unknown active_datasets: {unknown_names}")

        self.active_datasets = requested_names or list(self.dataset_names)
        self.global_index_map = []
        for dataset_name in self.dataset_names:
            dataset = self.datasets_by_name[dataset_name]
            dataset.set_stage(stage_cfg=stage_cfg)
            if dataset_name not in self.active_datasets:
                continue
            self.global_index_map.extend((dataset_name, idx) for idx in range(len(dataset)))
        self._len = len(self.global_index_map)
        return self

    def eval(self, mode="val", rank=0, world_size=1):
        dataset = self._get_eval_dataset()
        dataset.eval(mode=mode, rank=rank, world_size=world_size)
        self._sync_runtime_view(dataset)
        return self

    def get_selected_videos(self, mode: str | None = None) -> list[str]:
        if (self.mode if mode is None else mode) != "train":
            return self._get_eval_dataset().get_selected_videos(mode=mode)
        return []

    def get_trackeval_dataset_config(self, mode: str | None = None) -> dict:
        dataset = self._get_eval_dataset()
        if not hasattr(dataset, "get_trackeval_dataset_config"):
            return {}
        # codex : Preserve dataset-specific TrackEval overrides, such as MOT17 val_half
        # mapping to train/gt_val_half.txt during online evaluation of mixed training.
        return dataset.get_trackeval_dataset_config(mode=mode)

    def _get_eval_dataset(self) -> MOTDataset:
        return self.datasets_by_name[self.eval_dataset_name]

    def _restore_train_view(self):
        # codex : Keep the container state in train mode between online-eval passes.
        self.name = "MixedDataset"
        self.dataset_dir = ""
        self.mode = "train"
        self.video_name_prefix = self.name
        self.selected_videos = []
        self.split_dir = ""
        self.gts = {}
        self.vid_idx = {}
        self.idx_vid = {}
        self.transform = None
        self.sample_begin_frames = []
        self.sample_vid_max_frame = {}

    def _sync_runtime_view(self, dataset: MOTDataset):
        # codex : Reuse the primary dataset eval metadata so TrackEval sees the original dataset identity.
        self.name = dataset.name
        self.dataset_dir = dataset.dataset_dir
        self.mode = dataset.mode
        self.video_name_prefix = dataset.video_name_prefix
        self.eval_videos = dataset.eval_videos
        self.train_sub_sets = dataset.train_sub_sets
        self.selected_videos = dataset.selected_videos
        self.split_dir = dataset.split_dir
        self.gts = dataset.gts
        self.vid_idx = dataset.vid_idx
        self.idx_vid = dataset.idx_vid
        self.transform = dataset.transform
        self.batch_size = dataset.batch_size
        self.sample_length = dataset.sample_length
        self.sample_mode = dataset.sample_mode
        self.sample_interval = dataset.sample_interval
        self.sample_begin_frames = dataset.sample_begin_frames
        self.sample_vid_max_frame = dataset.sample_vid_max_frame
        self._len = len(dataset)

    @staticmethod
    def _normalize_active_dataset_names(active_datasets) -> list[str]:
        if isinstance(active_datasets, str):
            active_datasets = [active_datasets]
        normalized = []
        seen = set()
        for dataset_name in active_datasets:
            dataset_name = str(dataset_name).strip()
            if not dataset_name or dataset_name in seen:
                continue
            normalized.append(dataset_name)
            seen.add(dataset_name)
        return normalized
