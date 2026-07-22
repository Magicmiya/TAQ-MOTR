import os

from .dancetrack import DanceTrack


class SportsMOT(DanceTrack):

    def __init__(self, config: dict, mode: str, transform=None):
        dataset_cfg = dict(config)
        dataset_cfg.setdefault("dataset_dir", os.path.join(dataset_cfg["dataset_root"], "SportMOT", "dataset"))
        dataset_cfg.setdefault("video_name_prefix", "")
        super(SportsMOT, self).__init__(config=dataset_cfg, mode=mode, transform=transform)
        # SportsMOT stores train/val/test under SportMOT/dataset with sequence ids like v_xxx, so we normalize both here.


def build(config: dict, mode: str):
    return SportsMOT(config=config, mode=mode)
