import os
import queue
import shutil
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Optional

import cv2
import numpy as np
import torch
from bytecode import Bytecode, Instr


@dataclass
class HookEvent:
    name: str
    switch: str
    seq: int
    timestamp: float
    payload: dict[str, Any]


@dataclass
class FrameContext:
    mode: str
    video_name: str
    frame_id: int
    img_path: str
    image_bgr: Optional[np.ndarray]
    track_result: np.ndarray
    tag: Optional[str] = None
    _image_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_image_bgr(self) -> Optional[np.ndarray]:
        if self.image_bgr is not None:
            return self.image_bgr
        if not self.img_path:
            return None
        with self._image_lock:
            if self.image_bgr is None:
                img = cv2.imread(self.img_path)
                if img is None:
                    return None
                self.image_bgr = np.ascontiguousarray(img)
        return self.image_bgr


@dataclass
class TimeSession:
    device_index: Optional[int]
    sync_cuda: bool
    start_time: float


@dataclass
class ImageWriteJob:
    out_path: str
    image_bgr: np.ndarray


class GetTime:
    """Unified runtime timer with decorator and probe-style API."""

    enabled = False
    _switches: dict[str, bool] = {}
    _sessions: dict[str, list[TimeSession]] = defaultdict(list)
    _metrics: dict[str, dict[str, float | int]] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        name: Optional[str] = None,
        switch: Optional[str] = None,
        *,
        device: Optional[Any] = None,
        sync_cuda: bool = True,
    ):
        self.name = name
        self.switch = switch
        self.device = device
        self.sync_cuda = bool(sync_cuda)

    def __call__(self, func):
        timer_name = self.name or func.__qualname__
        timer_switch = self.switch or timer_name

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not type(self)._should_capture(timer_switch):
                return func(*args, **kwargs)

            device = self.device
            if device is None:
                device = type(self)._infer_device_from_call(args=args, kwargs=kwargs)

            type(self).pin(timer_name, mode="start", switch=timer_switch, device=device, sync_cuda=self.sync_cuda)
            try:
                return func(*args, **kwargs)
            finally:
                type(self).pin(timer_name, mode="end")

        return wrapper

    @classmethod
    def configure(cls, enabled: bool, cfg: Optional[dict[str, Any]] = None):
        cfg = cfg or {}
        cls.enabled = bool(enabled)
        cls._switches = {str(k): bool(v) for k, v in dict(cfg.get("switches", {})).items()}
        cls.clear()

    @classmethod
    def activate(cls):
        cls.enabled = True

    @classmethod
    def deactivate(cls):
        cls.enabled = False
        cls.clear()

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._sessions.clear()
            cls._metrics.clear()

    @classmethod
    def pin(
        cls,
        name: str,
        mode: str = "start",
        *,
        switch: Optional[str] = None,
        device: Optional[Any] = None,
        sync_cuda: bool = True,
    ):
        if not cls.enabled:
            return

        mode = mode.lower()
        if mode not in {"start", "end"}:
            raise ValueError(f"GetTime pin mode must be 'start' or 'end', got '{mode}'")

        if mode == "start":
            switch_key = switch or name
            if not cls._should_capture(switch_key):
                return
            cls._pin_start(name=name, device=device, sync_cuda=sync_cuda)
            return

        cls._pin_end(name=name)

    @classmethod
    @contextmanager
    def section(
        cls,
        name: str,
        *,
        switch: Optional[str] = None,
        device: Optional[Any] = None,
        sync_cuda: bool = True,
    ):
        cls.pin(name, mode="start", switch=switch, device=device, sync_cuda=sync_cuda)
        try:
            yield
        finally:
            cls.pin(name, mode="end")

    @classmethod
    def summary(cls) -> list[dict[str, float | int | str]]:
        with cls._lock:
            items = []
            for name, metric in sorted(cls._metrics.items()):
                count = int(metric["count"])
                total_s = float(metric["total_s"])
                avg_ms = (total_s * 1000.0 / count) if count > 0 else 0.0
                max_ms = float(metric["max_s"]) * 1000.0
                items.append(
                    {
                        "name": name,
                        "count": count,
                        "avg_ms": avg_ms,
                        "max_ms": max_ms,
                        "total_s": total_s,
                    }
                )
            return items

    @classmethod
    def _pin_start(cls, name: str, device: Optional[Any], sync_cuda: bool):
        device_index = cls._resolve_device(device)
        if sync_cuda and device_index is not None:
            cls._safe_sync_cuda(device_index)

        now = time.perf_counter()

        with cls._lock:
            sessions = cls._sessions[name]
            if sessions:
                session = sessions.pop()
                cls._close_session(name=name, session=session, end_time=now)
            sessions.append(
                TimeSession(
                    device_index=device_index,
                    sync_cuda=bool(sync_cuda),
                    start_time=now,
                )
            )

    @classmethod
    def _pin_end(cls, name: str):
        with cls._lock:
            sessions = cls._sessions.get(name)
            if not sessions:
                return
            session = sessions.pop()

        if session.sync_cuda and session.device_index is not None:
            cls._safe_sync_cuda(session.device_index)

        end_time = time.perf_counter()
        with cls._lock:
            cls._close_session(name=name, session=session, end_time=end_time)

    @classmethod
    def _close_session(cls, name: str, session: TimeSession, end_time: float):
        elapsed = max(0.0, end_time - session.start_time)
        metric = cls._metrics.setdefault(name, {"count": 0, "total_s": 0.0, "max_s": 0.0})
        metric["count"] = int(metric["count"]) + 1
        metric["total_s"] = float(metric["total_s"]) + elapsed
        metric["max_s"] = max(float(metric["max_s"]), elapsed)

    @classmethod
    def _should_capture(cls, switch: str) -> bool:
        if not cls.enabled:
            return False
        if "*" in cls._switches:
            return bool(cls._switches["*"])
        if switch in cls._switches:
            return bool(cls._switches[switch])
        return True

    @classmethod
    def _resolve_device(cls, device: Optional[Any]) -> Optional[int]:
        if not torch.cuda.is_available():
            return None

        if device is None:
            return None

        if isinstance(device, torch.device):
            if device.type != "cuda":
                return None
            return int(device.index) if device.index is not None else 0

        if isinstance(device, torch.Tensor):
            if not device.is_cuda:
                return None
            return int(device.device.index) if device.device.index is not None else 0

        try:
            return int(device)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _safe_sync_cuda(cls, device_index: int):
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize(device_index)
        except Exception:
            pass

    @classmethod
    def _infer_device_from_call(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[int]:
        if "frames" in kwargs:
            dev = cls._device_from_obj(kwargs.get("frames"))
            if dev is not None:
                return dev

        for obj in args:
            dev = cls._device_from_obj(obj)
            if dev is not None:
                return dev

        for obj in kwargs.values():
            dev = cls._device_from_obj(obj)
            if dev is not None:
                return dev

        return None

    @classmethod
    def _device_from_obj(cls, obj: Any) -> Optional[int]:
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                return int(obj.device.index) if obj.device.index is not None else 0
            return None

        tensor_attr = getattr(obj, "tensors", None)
        if isinstance(tensor_attr, torch.Tensor) and tensor_attr.is_cuda:
            return int(tensor_attr.device.index) if tensor_attr.device.index is not None else 0

        if isinstance(obj, (list, tuple)):
            for item in obj:
                dev = cls._device_from_obj(item)
                if dev is not None:
                    return dev

        if isinstance(obj, dict):
            for item in obj.values():
                dev = cls._device_from_obj(item)
                if dev is not None:
                    return dev

        return None


class TensorHook:
    """
    Unified hook decorator and event collector.

    Design notes:
    - Decorator only captures local variables and pushes immutable snapshots.
    - Queue payload keeps CPU data only so GPU memory can be released immediately after model forward.
    - Processing/statistics stays in visual tasks.
    """

    data: dict[str, list[dict[str, Any]]] = {}
    is_activate = False
    hard_off = False

    _queue_size = 2048
    _block_on_full = True
    _clone_tensor = True
    _switches: dict[str, bool] = {}
    _event_queue: queue.Queue[HookEvent] = queue.Queue(maxsize=_queue_size)
    _seq = 0
    _lock = threading.Lock()

    def __init__(self, keys: list[str], name: Optional[str] = None, switch: Optional[str] = None):
        self.keys = keys
        self.name = name
        self.switch = switch

    def __call__(self, func):
        if not getattr(func, "_tensor_hook_patched", False):
            c = Bytecode.from_code(func.__code__)
            load_code = []
            for key in self.keys:
                load_code += [
                    Instr("LOAD_FAST", key),
                    Instr("STORE_FAST", f"_visual_{key}"),
                ]
            extra_code = [Instr("STORE_FAST", "_res")] + load_code + [Instr("LOAD_FAST", "_res")]
            extra_code += [Instr("LOAD_FAST", f"_visual_{key}") for key in self.keys]
            extra_code += [
                Instr("BUILD_TUPLE", len(self.keys) + 1),
                Instr("STORE_FAST", "_result_tuple"),
                Instr("LOAD_FAST", "_result_tuple"),
            ]
            c[-1:-1] = extra_code
            func.__code__ = c.to_code()
            setattr(func, "_tensor_hook_patched", True)
            setattr(func, "_tensor_hook_num_keys", len(self.keys))

        hook_name = self.name or func.__qualname__
        hook_switch = self.switch or hook_name

        @wraps(func)
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            actual_res = res
            values = []
            if isinstance(res, tuple) and len(res) == int(getattr(func, "_tensor_hook_num_keys", 0)) + 1:
                actual_res = res[0]
                values = list(res[1:])

            if type(self)._should_capture(hook_switch):
                payload = self._build_payload(values)
                type(self)._push_event(
                    HookEvent(
                        name=hook_name,
                        switch=hook_switch,
                        seq=type(self)._next_seq(),
                        timestamp=time.time(),
                        payload=payload,
                    )
                )

            return actual_res

        return wrapper

    @classmethod
    def configure(cls, enabled: bool, cfg: Optional[dict[str, Any]] = None):
        cfg = cfg or {}
        cls.is_activate = bool(enabled)
        cls._block_on_full = bool(cfg.get("block_on_full", True))
        cls._clone_tensor = bool(cfg.get("clone_tensor", True))
        cls._switches = {str(k): bool(v) for k, v in dict(cfg.get("switches", {})).items()}

        queue_size = max(1, int(cfg.get("queue_size", cls._queue_size)))
        if queue_size != cls._queue_size:
            cls._queue_size = queue_size
            cls._event_queue = queue.Queue(maxsize=cls._queue_size)
        cls.clear()

    @classmethod
    def emit(cls, name: str, switch: str, payload: dict[str, Any]):
        if not cls._should_capture(switch):
            return
        snapshot = {k: cls._snapshot(v) for k, v in dict(payload).items()}
        cls._push_event(
            HookEvent(
                name=str(name),
                switch=str(switch),
                seq=cls._next_seq(),
                timestamp=time.time(),
                payload=snapshot,
            )
        )

    def _build_payload(self, values: list[Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for k, v in zip(self.keys, values):
            payload[k] = self._snapshot(v)
        return payload

    @classmethod
    def _snapshot(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.is_cuda:
                tensor = tensor.to(device="cpu", non_blocking=False)
            elif cls._clone_tensor:
                tensor = tensor.clone()
            if not cls._clone_tensor and not tensor.is_contiguous():
                tensor = tensor.contiguous()
            return tensor
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        if cls._is_track_instances_like(value):
            return value.result()
        if isinstance(value, list):
            return [cls._snapshot(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._snapshot(v) for v in value)
        if isinstance(value, dict):
            return {k: cls._snapshot(v) for k, v in value.items()}
        return value

    @staticmethod
    def _is_track_instances_like(value: Any) -> bool:
        if value is None:
            return False
        cls_name = value.__class__.__name__
        if cls_name != "TrackInstances":
            return False
        result_fn = getattr(value, "result", None)
        return callable(result_fn)

    @classmethod
    def _next_seq(cls) -> int:
        with cls._lock:
            cls._seq += 1
            return cls._seq

    @classmethod
    def _should_capture(cls, switch: str) -> bool:
        if cls.hard_off or (not cls.is_activate):
            return False
        if "*" in cls._switches:
            return bool(cls._switches["*"])
        return bool(cls._switches.get(switch, False))

    @classmethod
    def _push_event(cls, event: HookEvent):
        if cls._block_on_full:
            cls._event_queue.put(event, block=True)
            return
        try:
            cls._event_queue.put_nowait(event)
        except queue.Full:
            _ = cls._event_queue.get_nowait()
            cls._event_queue.put_nowait(event)

    @classmethod
    def drain(cls) -> list[HookEvent]:
        items: list[HookEvent] = []
        while True:
            try:
                items.append(cls._event_queue.get_nowait())
            except queue.Empty:
                break
        return items

    @classmethod
    def clear(cls):
        while True:
            try:
                cls._event_queue.get_nowait()
            except queue.Empty:
                break

    @classmethod
    def activate(cls):
        cls.is_activate = True

    @classmethod
    def deactivate(cls):
        cls.is_activate = False
        cls.clear()

    @classmethod
    def result(cls):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in cls.drain():
            grouped.setdefault(event.name, []).append(event.payload)
        return grouped


class ImageWriterPool:
    def __init__(self, num_workers: int, queue_size: int):
        self.num_workers = max(1, int(num_workers))
        self.queue_size = max(1, int(queue_size))
        self._queue: queue.Queue[Optional[ImageWriteJob]] = queue.Queue(maxsize=self.queue_size)
        self._workers: list[threading.Thread] = []
        self._closed = False

        for idx in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"visual-writer-{idx}", daemon=True)
            worker.start()
            self._workers.append(worker)

    def submit(self, job: ImageWriteJob):
        if self._closed:
            raise RuntimeError("ImageWriterPool is already closed.")
        self._queue.put(job, block=True)

    def close(self):
        if self._closed:
            return
        self._queue.join()
        for _ in self._workers:
            self._queue.put(None, block=True)
        for worker in self._workers:
            worker.join()
        self._closed = True

    def _worker_loop(self):
        while True:
            item = self._queue.get(block=True)
            if item is None:
                self._queue.task_done()
                break
            os.makedirs(os.path.dirname(item.out_path), exist_ok=True)
            cv2.imwrite(item.out_path, item.image_bgr)
            self._queue.task_done()


class BaseVisualTask:
    def __init__(self, task_name: str, cfg: dict[str, Any], mode: str, root_dir: str):
        self.task_name = task_name
        self.cfg = cfg
        self.mode = mode
        self.root_dir = root_dir
        self.enabled = bool(cfg.get("enabled", False))
        self._image_writer: Optional[ImageWriterPool] = None

    def bind_image_writer(self, writer: Optional[ImageWriterPool]):
        self._image_writer = writer

    def required_switches(self) -> set[str]:
        return set()

    def required_time_switches(self) -> set[str]:
        return set()

    def requires_image(self) -> bool:
        return False

    def init(self):
        pass

    def update(self, frame: FrameContext, hook_events: list[HookEvent]):
        pass

    def close(self):
        pass

    def video_dir(self, video_name: str) -> str:
        out_dir = os.path.join(self.root_dir, video_name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def task_dir(self, video_name: str) -> str:
        out_dir = os.path.join(self.video_dir(video_name), self.task_name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    @staticmethod
    def frame_filename(frame: FrameContext, suffix: str = ".jpg") -> str:
        return f"{frame.tag}{suffix}" if frame.tag else f"{frame.frame_id:06d}{suffix}"

    def submit_image(self, frame: FrameContext, image_bgr: np.ndarray, suffix: str = ".jpg"):
        out_path = os.path.join(self.task_dir(frame.video_name), self.frame_filename(frame, suffix=suffix))
        if self._image_writer is None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, image_bgr)
            return
        self._image_writer.submit(ImageWriteJob(out_path=out_path, image_bgr=image_bgr))


class Visualizer:
    """Dispatcher that routes frame updates to independent visual tasks."""

    def __init__(self, cfg: dict, mode: str, save_path: str):
        self.mode = mode
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.save_path = save_path
        root_dir_name = str(cfg.get("root_dir_name", "visualize"))
        output_root_dir = cfg.get("output_root_dir", "")
        if output_root_dir:
            output_root_dir = str(output_root_dir)
            base_root_dir = output_root_dir if os.path.isabs(output_root_dir) else os.path.join(save_path, output_root_dir)
        else:
            base_root_dir = save_path
        # Keep per-checkpoint output isolation while allowing the runtime pipeline to be internally opinionated.
        self.root_dir = os.path.join(base_root_dir, root_dir_name)

        self.num_workers = max(1, int(cfg.get("workers", self._default_worker_count())))
        self.writer_workers = max(1, int(cfg.get("writer_workers", 2)))
        self.frame_queue_size = max(128, self.num_workers * 32)
        self.writer_queue_size = max(256, self.writer_workers * 64)
        self._frame_queue: queue.Queue[Optional[tuple[FrameContext, list[HookEvent]]]] = queue.Queue(
            maxsize=self.frame_queue_size
        )
        self._frame_workers: list[threading.Thread] = []
        self._image_writer: Optional[ImageWriterPool] = None
        self._closed = False

        if self.enabled:
            self._prepare_root()

        self.tasks = self._build_tasks(cfg)
        self._need_image = any(task.requires_image() for task in self.tasks if task.enabled)

        if self.enabled:
            self._image_writer = ImageWriterPool(num_workers=self.writer_workers, queue_size=self.writer_queue_size)
        for task in self.tasks:
            task.bind_image_writer(self._image_writer)
            task.init()

        hook_cfg = dict(cfg.get("hook", {}))
        hook_switches = {str(k): bool(v) for k, v in dict(hook_cfg.get("switches", {})).items()}
        for task in self.tasks:
            if task.enabled:
                for switch in task.required_switches():
                    hook_switches.setdefault(str(switch), True)
        hook_cfg["switches"] = hook_switches

        hook_enabled = bool(hook_cfg.get("enabled", False)) and self.enabled
        TensorHook.configure(enabled=hook_enabled, cfg=hook_cfg)

        time_cfg = dict(cfg.get("time", {}))
        time_switches = {str(k): bool(v) for k, v in dict(time_cfg.get("switches", {})).items()}
        for task in self.tasks:
            if task.enabled:
                for switch in task.required_time_switches():
                    time_switches.setdefault(str(switch), True)
        time_cfg["switches"] = time_switches

        time_enabled = bool(time_cfg.get("enabled", False)) and self.enabled
        GetTime.configure(enabled=time_enabled, cfg=time_cfg)

        if self.enabled:
            for idx in range(self.num_workers):
                worker = threading.Thread(target=self._frame_worker_loop, name=f"visual-frame-{idx}", daemon=True)
                worker.start()
                self._frame_workers.append(worker)

    @staticmethod
    def _default_worker_count() -> int:
        cpu_count = os.cpu_count() or 1
        return max(1, min(4, cpu_count))

    def _prepare_root(self):
        if os.path.isdir(self.root_dir):
            shutil.rmtree(self.root_dir)
        os.makedirs(self.root_dir, exist_ok=True)

    def _build_tasks(self, cfg: dict) -> list[BaseVisualTask]:
        from .tasks.bbox_render import BBoxRenderTask
        from .tasks.decoder_l0_query_focus import DecoderL0QueryFocusTask
        from .tasks.det_recover_monitor import DetRecoverMonitorTask
        from .tasks.grad_monitor import GradMonitorTask
        from .tasks.hqg_histogram import HQGHistogramTask
        from .tasks.hqg_topk_roi_map import HQGTopKRoiMapTask
        from .tasks.runtime_profile import RuntimeProfileTask

        tasks_cfg = dict(cfg.get("tasks", {}))
        task_list: list[BaseVisualTask] = []

        bbox_cfg = dict(tasks_cfg.get("bbox_render", {}))
        task_list.append(BBoxRenderTask("bbox_render", bbox_cfg, self.mode, self.root_dir))

        hqg_roi_cfg = dict(tasks_cfg.get("hqg_topk_roi_map", {}))
        # Keep ROI-map rendering independent from bbox export so each task can scale with the shared worker pool.
        task_list.append(HQGTopKRoiMapTask("hqg_topk_roi_map", hqg_roi_cfg, self.mode, self.root_dir))

        hqg_cfg = dict(tasks_cfg.get("hqg_histogram", {}))
        task_list.append(HQGHistogramTask("hqg_histogram", hqg_cfg, self.mode, self.root_dir))

        focus_cfg = dict(tasks_cfg.get("decoder_l0_query_focus", {}))
        # Decoder-layer focus visualization is kept separate from HQG ROI maps because it explains a different stage.
        task_list.append(DecoderL0QueryFocusTask("decoder_l0_query_focus", focus_cfg, self.mode, self.root_dir))

        recover_cfg = dict(tasks_cfg.get("det_recover_monitor", {}))
        task_list.append(DetRecoverMonitorTask("det_recover_monitor", recover_cfg, self.mode, self.root_dir))

        runtime_cfg = dict(tasks_cfg.get("runtime_profile", {}))
        if "enabled" not in runtime_cfg:
            runtime_cfg["enabled"] = bool(dict(cfg.get("time", {})).get("enabled", False))
        task_list.append(RuntimeProfileTask("runtime_profile", runtime_cfg, self.mode, self.root_dir))

        grad_cfg = dict(tasks_cfg.get("grad_monitor", {}))
        if not self.enabled:
            grad_cfg["enabled"] = False
        # Gradient diagnostics are a visualizer-managed task, but their TensorBoard log lives beside train logs.
        task_list.append(GradMonitorTask("grad_monitor", grad_cfg, self.mode, self.save_path))

        return task_list

    def before_grad_clip(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
    ):
        if not self.enabled:
            return
        for task in self.tasks:
            if task.enabled and hasattr(task, "before_grad_clip"):
                task.before_grad_clip(model=model, optimizer=optimizer, epoch=epoch, global_step=global_step)

    def after_grad_clip(
        self,
        pre_clip_norm: torch.Tensor | float | None,
        max_norm: float,
        global_step: int,
    ):
        if not self.enabled:
            return
        for task in self.tasks:
            if task.enabled and hasattr(task, "after_grad_clip"):
                task.after_grad_clip(pre_clip_norm=pre_clip_norm, max_norm=max_norm, global_step=global_step)

    def compute_global_grad_norm(self, model: torch.nn.Module) -> torch.Tensor | None:
        if not self.enabled:
            return None
        for task in self.tasks:
            if task.enabled and hasattr(task, "compute_global_grad_norm"):
                norm = task.compute_global_grad_norm(model=model)
                if norm is not None:
                    return norm
        return None

    def update(
        self,
        batch: object,
        track_result: np.ndarray = np.empty((0, 9), dtype=np.float32),
        index: np.ndarray = np.empty((0,), dtype=np.int32),
        tag: Optional[str] = None,
    ):
        del index

        if not self.enabled:
            TensorHook.clear()
            return

        info0 = self._get_info0(batch)
        if info0 is None:
            TensorHook.clear()
            return

        frame_id = info0.get("frame_idx", 0)
        frame_id = int(frame_id.item() if hasattr(frame_id, "item") else frame_id)
        video_name = str(info0.get("video_name", "unknown_video"))
        img_path = str(info0.get("img_path", ""))

        events = TensorHook.drain()
        frame = FrameContext(
            mode=self.mode,
            video_name=video_name,
            frame_id=frame_id,
            img_path=img_path,
            image_bgr=None,
            track_result=np.array(track_result, copy=True),
            tag=tag,
        )
        self._frame_queue.put((frame, events), block=True)

    def _frame_worker_loop(self):
        while True:
            item = self._frame_queue.get(block=True)
            if item is None:
                self._frame_queue.task_done()
                break
            frame, events = item
            self._process_frame(frame, events)
            self._frame_queue.task_done()

    def _process_frame(self, frame: FrameContext, events: list[HookEvent]):
        if self._need_image:
            frame.get_image_bgr()
        for task in self.tasks:
            if task.enabled:
                task.update(frame, events)

    def show(self, delay: int = 0):
        del delay
        return

    def close(self):
        if self._closed:
            return

        if self.enabled:
            self._frame_queue.join()
            for _ in self._frame_workers:
                self._frame_queue.put(None, block=True)
            for worker in self._frame_workers:
                worker.join()
            self._frame_workers = []

            if self._image_writer is not None:
                self._image_writer.close()
                self._image_writer = None

        for task in self.tasks:
            if task.enabled:
                task.close()

        TensorHook.clear()
        self._closed = True

    @staticmethod
    def _get_info0(batch: object) -> Optional[dict]:
        infos = None
        frame_slot = None

        if hasattr(batch, "infos_cpu") or hasattr(batch, "infos"):
            infos = getattr(batch, "infos_cpu", None) or getattr(batch, "infos", None)
            frame_slot = getattr(batch, "frame_idx", None)
        elif isinstance(batch, dict) and "infos" in batch:
            infos = batch.get("infos")
            frame_slot = 0

        if infos is None:
            return None

        if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], (list, tuple)):
            if frame_slot is None:
                frame_slot = 0
            infos = infos[int(frame_slot)]

        if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
            return infos[0]
        if isinstance(infos, dict):
            return infos
        return None


__all__ = [
    "TensorHook",
    "GetTime",
    "Visualizer",
    "BaseVisualTask",
    "HookEvent",
    "FrameContext",
]
