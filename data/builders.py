from __future__ import annotations

from copy import deepcopy

from .MOTChallenge import build as build_motchallenge
from .SportsMOT import build as build_sportsmot
from .dancetrack import build as build_dancetrack
from .image_dataset import build as build_image_dataset


DATASET_BUILDERS = {
    "DanceTrack": build_dancetrack,
    "SportsMOT": build_sportsmot,
    "MOT17": build_motchallenge,
    "MOT20": build_motchallenge,
    "CrowdHuman": build_image_dataset,
    "CityPersons": build_image_dataset,
}


def build_named_dataset(config: dict, mode: str):
    dataset_name = str(config["dataset_name"]).strip()
    if dataset_name not in DATASET_BUILDERS:
        raise ValueError(f"Dataset {dataset_name} is not supported!")
    return DATASET_BUILDERS[dataset_name](config=config, mode=mode)


def normalize_train_dataset_configs(config: dict) -> list[dict]:
    raw_train_datasets = config.get("train_datasets", None)
    if raw_train_datasets is None:
        raw_train_datasets = config.get("dataset_name")

    if isinstance(raw_train_datasets, str):
        dataset_cfg = deepcopy(config)
        dataset_cfg["dataset_name"] = raw_train_datasets
        dataset_cfg["train_sub_sets"] = config.get("train_sub_sets", [])
        return [dataset_cfg]

    if not isinstance(raw_train_datasets, dict) or not raw_train_datasets:
        raise TypeError("Dataset.dataset_name or Dataset.train_datasets must be a non-empty string or mapping.")

    dataset_cfgs = []
    for dataset_name, train_sub_sets in raw_train_datasets.items():
        dataset_cfg = deepcopy(config)
        dataset_cfg["dataset_name"] = str(dataset_name).strip()
        dataset_cfg["train_sub_sets"] = train_sub_sets
        dataset_cfgs.append(dataset_cfg)
    return dataset_cfgs
