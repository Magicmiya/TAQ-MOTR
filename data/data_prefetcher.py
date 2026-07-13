import warnings
from collections.abc import MutableMapping, Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch


@dataclass
class PrefetchedFrame:
    """Container describing a single frame that has been moved to the GPU."""

    frame_idx: int
    imgs: Any
    infos: Any
    infos_cpu: Any | None


class PrefetchedBatch(MutableMapping):
    """
    Wraps a raw dataloader batch and exposes streaming helpers.

    The mapping interface (`batch["imgs"]`, `batch["infos"]`, etc.) behaves just
    like the original dataloader output so existing training code can keep
    working. Additional helpers (`preload`, `next_frame`, `has_next_frame`)
    provide a lightweight streaming API:

        batch = next(prefetcher)
        frame = batch.next_frame()      # wait for already prefetched frame
        ...                             # run model on `frame.imgs`
        batch.preload()                 # schedule next frame transfer
    """

    def __init__(
        self,
        batch: dict[str, Any],
        only_imgs: bool,
        on_depleted=None,
        auto_preload: bool = False,
    ):
        self._data = dict(batch)
        self.only_imgs = only_imgs
        self.stream = cast(torch.cuda.Stream,torch.cuda.Stream())
        self._on_depleted = on_depleted
        self._depleted_notified = False
        self.auto_preload = auto_preload

        self._sequence_len = len(self._data["imgs"])
        self._next_frame_idx = 0
        self._pending: PrefetchedFrame | None = None
        self.infos_gpu = None
        self._preload_requested = False

        if self.only_imgs:
            with torch.cuda.stream(self.stream):
                self.infos_gpu = _recursive_to_cuda(self._data["infos"])

        self.preload()  # Prime the first frame so next_frame() can be called immediately.

    # --- Mapping interface -------------------------------------------------
    def __getitem__(self, key):
        if key =="infos" and self.infos_gpu is not None:
            return self.infos_gpu
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return self

    def __next__(self):
        frame = self.next_frame()
        if frame is None:
            raise StopIteration
        return frame

    def __len__(self):
        return len(self._data)

    # --- Streaming helpers -------------------------------------------------
    def preload(self):
        """
        Schedule a non-blocking transfer of the next frame if no other transfer
        is in-flight. Safe to call multiple times; redundant calls are ignored.
        """
        if self._pending is not None:
            return
        if self._next_frame_idx >= self._sequence_len:
            self._notify_depleted()
            return

        frame_idx = self._next_frame_idx
        self._next_frame_idx += 1

        frame_imgs = self._data["imgs"][frame_idx]
        frame_infos = self._data["infos"][frame_idx]

        with torch.cuda.stream(self.stream):
            frame_imgs_gpu = _recursive_to_cuda(frame_imgs)
            frame_infos_gpu = None if self.only_imgs else _recursive_to_cuda(frame_infos)

        infos_payload = self.infos_gpu if self.only_imgs else frame_infos_gpu
        self._pending = PrefetchedFrame(
            frame_idx=frame_idx,
            imgs=frame_imgs_gpu,
            infos=infos_payload,
            infos_cpu=frame_infos if self.only_imgs else None
        )
        self._preload_requested = True

    def next_frame(self) -> PrefetchedFrame | None:
        """
        Wait for the in-flight transfer to finish and return the prefetched frame.
        Automatically clears the pending slot; callers are expected to invoke
        ``preload()`` afterwards to start the next transfer.
        """
        if self._pending is None:
            if self._next_frame_idx >= self._sequence_len:
                self._notify_depleted()
                return None
            if not self._preload_requested:
                raise RuntimeError(
                    "PrefetchedBatch.next_frame() called before preload(). "
                    "Invoke batch.preload() after consuming each frame."
                )
            raise RuntimeError("PrefetchedBatch internal state inconsistent: preload requested but no pending frame.")

        torch.cuda.current_stream().wait_stream(self.stream)
        _record_stream(self._pending.imgs)
        if self._pending.infos is not None:
            _record_stream(self._pending.infos)
        frame = self._pending
        self._pending = None
        self._preload_requested = False

        if self._next_frame_idx >= self._sequence_len:
            self._notify_depleted()
        elif self.auto_preload:
            self.preload()

        return frame

    def has_next_frame(self) -> bool:
        """Return True if there are still frames that can be streamed."""
        return self._pending is not None or self._next_frame_idx < self._sequence_len

    def batch_size(self) -> int:
        infos = self._data["infos"]
        assert isinstance(infos, (list, tuple)) and len(infos) > 0, "batch infos must be a non-empty list/tuple."
        first_frame_infos = infos[0]
        assert isinstance(first_frame_infos, (list, tuple)), "frame infos must be a list/tuple."
        return len(first_frame_infos)

    def get_info(self, frame_idx: int = 0, sample_idx: int = 0) -> dict[str, Any]:
        infos = self._data["infos"]
        assert isinstance(infos, (list, tuple)), "batch infos must be a list/tuple."
        assert 0 <= frame_idx < len(infos), f"frame_idx out of range: {frame_idx}"
        frame_infos = infos[frame_idx]
        assert isinstance(frame_infos, (list, tuple)), "frame infos must be a list/tuple."
        assert 0 <= sample_idx < len(frame_infos), f"sample_idx out of range: {sample_idx}"
        info = frame_infos[sample_idx]
        assert isinstance(info, dict), f"frame info must be dict, got {type(info)}"
        return info

    def get_video_name(self, frame_idx: int = 0, sample_idx: int = 0) -> str:
        info = self.get_info(frame_idx=frame_idx, sample_idx=sample_idx)
        video_name = info.get("video_name")
        assert isinstance(video_name, str), f"video_name must be str, got {type(video_name)}"
        return video_name

    def get_frame_idx(self, frame_idx: int = 0, sample_idx: int = 0) -> int:
        info = self.get_info(frame_idx=frame_idx, sample_idx=sample_idx)
        frame_value = info["frame_idx"]
        if not isinstance(frame_value, torch.Tensor):
            raise TypeError(f"frame_idx must be torch.Tensor, got {type(frame_value)}")
        return int(frame_value.item())

    def _notify_depleted(self):
        if self._depleted_notified:
            return
        self._depleted_notified = True
        if callable(self._on_depleted):
            self._on_depleted()



class DataPrefetcher:
    """
    Wraps a dataloader and returns :class:`PrefetchedBatch` objects instead of
    raw tuples. Each batch keeps CPU data accessible while exposing helpers to
    move individual frames to the GPU without blowing up peak memory.
    """

    def __init__(self, loader, only_imgs: bool = True, auto_preload_frames: bool = False):
        """
        Args:
            loader: Iterable whose batches contain at least ``imgs`` and
                ``infos`` entries as described in the README.
            only_imgs: If True, frame metadata is copied to GPU exactly once
                per batch and reused for every streamed frame.
        """
        self._loader = loader
        self.loader = iter(loader)
        self.only_imgs = only_imgs
        self.auto_preload_frames = auto_preload_frames

        self._validated_format = False
        self._active_batch: PrefetchedBatch | None = None
        self._staged_batch: PrefetchedBatch | None = None

        self._check_pin_memory(loader)
        initial_batch = self._fetch_raw_batch()
        if initial_batch is not None:
            self._active_batch = self._build_prefetched_batch(initial_batch)

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        return self

    def __next__(self) -> PrefetchedBatch:
        if self._active_batch is None:
            raise StopIteration

        batch = self._active_batch
        if self._staged_batch is not None:
            self._active_batch = self._staged_batch
            self._staged_batch = None
        else:
            next_raw = self._fetch_raw_batch()
            self._active_batch = self._build_prefetched_batch(next_raw) if next_raw is not None else None
        return batch

    def _build_prefetched_batch(self, batch: dict[str, Any]) -> PrefetchedBatch:
        return PrefetchedBatch(
            batch=batch,
            only_imgs=self.only_imgs,
            on_depleted=self._schedule_next_batch,
            auto_preload=self.auto_preload_frames
        )

    def _schedule_next_batch(self):
        if self._staged_batch is not None:
            return
        next_raw = self._fetch_raw_batch()
        if next_raw is None:
            self._staged_batch = None
            return
        self._staged_batch = self._build_prefetched_batch(next_raw)

    def _fetch_raw_batch(self) -> dict[str, Any] | None:
        try:
            raw_batch = next(self.loader)
        except StopIteration:
            return None

        normalized = self._normalize_batch(raw_batch)
        if not self._validated_format:
            self._validate_batch(normalized["imgs"], normalized["infos"])
            self._validated_format = True
        return normalized

    @staticmethod
    def _normalize_batch(batch) -> dict[str, Any]:
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, (list, tuple)):
            assert len(batch) == 2, (
                "DataLoader output must be a mapping or a 2-item tuple/list."
            )
            return {"imgs": batch[0], "infos": batch[1]}
        raise TypeError(
            f"Unsupported batch type {type(batch)}; expected mapping or tuple/list."
        )

    @staticmethod
    def _check_pin_memory(loader):
        pin_memory_flag = getattr(loader, "pin_memory", None)
        if pin_memory_flag is None:
            return
        if not pin_memory_flag:
            warnings.warn(
                "DataLoader pin_memory is disabled; enabling it typically improves "
                "transfer overlap for DataPrefetcher.",
                stacklevel=2,
            )

    @staticmethod
    def _validate_batch(batch_imgs, batch_infos):
        assert isinstance(batch_imgs, (list, tuple)), (
            "Expected batch_imgs to be a list/tuple of per-frame tensors, "
            f"got {type(batch_imgs)}"
        )
        assert len(batch_imgs) > 0, "batch_imgs must contain at least one frame."
        first_img = batch_imgs[0]
        assert isinstance(first_img, torch.Tensor), (
            "Each frame in batch_imgs must be a torch.Tensor, "
            f"got {type(first_img)}"
        )
        assert isinstance(batch_infos, (list, tuple)), (
            "Expected batch_infos to be a list/tuple of per-frame metadata, "
            f"got {type(batch_infos)}"
        )
        assert len(batch_infos) == len(batch_imgs), (
            "batch_infos length must match batch_imgs length "
            f"({len(batch_infos)} vs {len(batch_imgs)})."
        )


def _recursive_to_cuda(data):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cuda(non_blocking=True)
    if isinstance(data, list):
        return [_recursive_to_cuda(item) for item in data]
    if isinstance(data, tuple):
        return tuple(_recursive_to_cuda(item) for item in data)
    if isinstance(data, dict):
        return {k: _recursive_to_cuda(v) for k, v in data.items()}
    return data


def _record_stream(data):
    if data is None:
        return
    if isinstance(data, torch.Tensor):
        data.record_stream(torch.cuda.current_stream())
        return
    if isinstance(data, (list, tuple)):
        for item in data:
            _record_stream(item)
        return
    if isinstance(data, dict):
        for value in data.values():
            _record_stream(value)
