import itertools
import random

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler

from .baseDataset import MOTDataset
from utils.utils import dist_rank, is_dist


class MOTBatchSampler:
    def __init__(
        self,
        sampler,
        batch_size: int,
        drop_last: bool = False,
        skip_batches: int = 0,
        dataset: MOTDataset | None = None,
        seed: int = 0,
    ):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.skip_batches = max(0, int(skip_batches))
        self.dataset = dataset
        self.seed = int(seed)

    def _chunk_indices(self, indices: list[int]) -> list[list[int]]:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        batches = [indices[start:start + self.batch_size] for start in range(0, len(indices), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        return batches

    def __iter__(self):
        batches = self._chunk_indices(list(self.sampler))
        batch_iter = enumerate(batches)
        if self.skip_batches > 0:
            batch_iter = itertools.islice(batch_iter, self.skip_batches, None)
        for batch_idx, batch in batch_iter:
            # codex : Derive random_idx from batch position so mid-epoch resume matches uninterrupted training.
            random_idx = random.Random(self.seed + batch_idx).randint(0, 12)
            yield [(random_idx, idx) for idx in batch]

    def __len__(self):
        base_len = len(self.sampler)
        if self.drop_last:
            total_batches = base_len // self.batch_size
        else:
            total_batches = (base_len + self.batch_size - 1) // self.batch_size
        return max(total_batches - self.skip_batches, 0)


def build_sampler(dataset: MOTDataset, shuffle=False, epoch=int(0), is_eval=False, seed: int = 0) -> Sampler:
    if is_eval:
        sampler = SequentialSampler(dataset)
    else:
        if is_dist():
            sampler = DistributedSampler(dataset=dataset, shuffle=shuffle, seed=int(seed))
            sampler.set_epoch(epoch)
        else:
            generator = torch.Generator()
            generator.manual_seed(int(seed) + int(epoch))
            sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
    return sampler


def build_batch_sampler_seed(base_seed: int, epoch: int) -> int:
    # codex : Rank-specific batch seeds avoid identical frame-offset choices across DDP ranks.
    return int(base_seed) + int(epoch) * 100000 + dist_rank() * 1000
