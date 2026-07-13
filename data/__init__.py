from .baseDataset import MOTDataset
from .builders import build_named_dataset
from .data_prefetcher import DataPrefetcher, PrefetchedBatch
from .dataloader import build_dataloader
from .mixed_dataset import MixedDataset


def build_dataset(config: dict, mode: str) -> MOTDataset:
    if mode == "train":
        return MixedDataset(config=config, mode=mode)

    if not isinstance(config.get("dataset_name"), str):
        raise TypeError("Eval/Test dataset_name must stay a single dataset string.")
    return build_named_dataset(config=config, mode=mode)
