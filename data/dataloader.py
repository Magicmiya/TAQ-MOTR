# BatchDataLoader ref from Megvii Yolox https://github.com/Megvii-BaseDetection/YOLOX
import random
from functools import partial
import torch.nn.functional as F
from typing import Optional, Type, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from .baseDataset import MOTDataset
from .sampler import MOTBatchSampler, build_batch_sampler_seed, build_sampler
from utils.utils import dist_rank


def _seed_worker(worker_id: int, base_seed: int, epoch: int):
    worker_base_seed = int(base_seed) + int(epoch) * 100000 + dist_rank() * 1000
    # Lock Python/NumPy/Torch RNGs inside workers for reproducible augmentation.
    worker_seed = worker_base_seed + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def build_dataloader(
    dataset: MOTDataset,
    config: dict,
    sampler: Optional[Sampler] = None,
    skip_batches: int = 0,
    seed: int = 0,
) -> DataLoader:
    seed = int(seed)
    if sampler is None:
        sampler = build_sampler(
            dataset=dataset,
            shuffle=config['sampler_shuffle'],
            epoch=dataset.epoch,
            is_eval=dataset.mode != 'train',
            seed=seed,
        )
    generator = torch.Generator()
    generator.manual_seed(seed + int(dataset.epoch) * 100000 + dist_rank() * 1000)
    return DataLoader(
        dataset=dataset,
        batch_sampler=MOTBatchSampler(
            sampler,
            dataset.batch_size,
            drop_last=False if 'drop_last' not in config.keys() else config['drop_last'],
            skip_batches=skip_batches,
            dataset=dataset,
            seed=build_batch_sampler_seed(seed, dataset.epoch),
        ),
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=config['num_workers'],
        persistent_workers=config['persistent_workers'],
        prefetch_factor=config['prefetch_factor'],
        worker_init_fn=partial(_seed_worker, base_seed=seed, epoch=dataset.epoch),
        generator=generator,
    )


def collate_fn(samples) -> dict[str, Any]:
    seq_len = len(samples[0]['imgs'])
    imgs, infos = [], []
    for seq_len in range(seq_len):
        batch_imgs, batch_infos = [], []
        for sample in samples:
            batch_imgs.append(sample['imgs'][seq_len])
            batch_infos.append(sample['infos'][seq_len])
        imgs.append(torch.stack(batch_imgs))
        infos.append(batch_infos)
    return {'imgs': imgs, 'infos': infos}
