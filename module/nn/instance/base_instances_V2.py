# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)

import torch
from collections import OrderedDict
from typing import Optional, Union, Dict, List, MutableMapping, Any, TypeVar
from ..common import logits_to_scores

# Preserve subclass type through factory-like methods for static typing.
T_Instances = TypeVar("T_Instances", bound="BaseInstances")


class BaseInstances(object):
    """
    Track the base class of the instance.

    This class provides the basic structure required for tracking instances to manage the target data in each frame of
    the image.Data management allows instances to be sliced and indexed like Tensors, and when there is no target,
    it enables data output of length 0 to ensure the shape of Tensor-type data
    """

    DEFAULT_HISTORY_LEN = 16

    _FIELD_REGISTRY: OrderedDict[str, Any] = OrderedDict(
        [
            (
                "query_embed",
                {"size": lambda self: (self._len, self.hidden_dim), "dtype": torch.float32, "fill": 0.0, "lazy": True},
            ),
            ("ref_pts", {"size": lambda self: (self._len, 4), "dtype": torch.float32, "fill": 0.5}),
            ("ids", {"size": lambda self: (self._len,), "dtype": torch.long, "fill": -1}),
            ("states", {"size": lambda self: (self._len,), "dtype": torch.long, "fill": 4}),
            ("matched_idx", {"size": lambda self: (self._len,), "dtype": torch.long, "fill": -1}),
            ("boxes", {"size": lambda self: (self._len, 4), "dtype": torch.float32, "fill": 0.5}),
            ("logits", {"size": lambda self: (self._len, self.num_classes), "dtype": torch.float32, "fill": 0.0}),
            (
                "output_embed",
                {"size": lambda self: (self._len, self.hidden_dim), "dtype": torch.float32, "fill": 0.0, "lazy": True},
            ),
            (
                "last_output",
                {"size": lambda self: (self._len, self.hidden_dim), "dtype": torch.float32, "fill": 0.0, "lazy": True},
            ),
            (
                "long_memory",
                {"size": lambda self: (self._len, self.hidden_dim), "dtype": torch.float32, "fill": 0.0, "lazy": True},
            ),
            ("last_appear_boxes", {"size": lambda self: (self._len, 4), "dtype": torch.float32, "fill": 0.5}),
            ("disappear_time", {"size": lambda self: (self._len,), "dtype": torch.long, "fill": -1}),
            ("labels", {"size": lambda self: (self._len,), "dtype": torch.long, "fill": -1}),
            ("iou", {"size": lambda self: (self._len,), "dtype": torch.float32, "fill": 0.0}),
            (
                "hist_boxes",
                {"size": lambda self: (self._len, self.history_len, 4), "dtype": torch.float32, "fill": 0.0},
            ),
            (
                "hist_frame_idx",
                {"size": lambda self: (self._len, self.history_len), "dtype": torch.long, "fill": -1},
            ),
            (
                "hist_output_embed",
                {
                    "size": lambda self: (self._len, self.history_len, self.hidden_dim),
                    "dtype": torch.float32,
                    "fill": 0.0,
                },
            ),
            ("hist_ptr", {"size": lambda self: (self._len,), "dtype": torch.long, "fill": 0}),
            ("hist_count", {"size": lambda self: (self._len,), "dtype": torch.long, "fill": 0}),
        ]
    )
    _tensor_attrs = tuple(_FIELD_REGISTRY.keys())
    _lazy_tensor_attrs = tuple(name for name, spec in _FIELD_REGISTRY.items() if spec.get("lazy", False))

    frame_height: float
    frame_width: float
    hidden_dim: int
    num_classes: int
    frame: int
    _len: int
    device: torch.device
    history_len: int

    query_embed: torch.Tensor  # init & update by the penultimate layer content
    ref_pts: torch.Tensor  # init & update by the penultimate layer position
    ids: torch.Tensor
    states: torch.Tensor
    boxes: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    output_embed: torch.Tensor  # init & update by the last layer content

    # for life-cycle management
    last_output: torch.Tensor  # init & update by the last layer content
    long_memory: torch.Tensor  # init & update by the penultimate layer content
    last_appear_boxes: torch.Tensor
    disappear_time: torch.Tensor
    hist_boxes: torch.Tensor
    hist_frame_idx: torch.Tensor
    hist_output_embed: torch.Tensor
    hist_ptr: torch.Tensor
    hist_count: torch.Tensor

    # for Training
    matched_idx: torch.Tensor
    iou: torch.Tensor

    def __init__(
        self,
        frame_height: float,
        frame_width: float,
        hidden_dim: int,
        num_classes: int,
        frame: int,
        device: torch.device,
        **kwargs,
    ):
        object.__setattr__(self, "frame_height", frame_height)
        object.__setattr__(self, "frame_width", frame_width)
        object.__setattr__(self, "hidden_dim", hidden_dim)
        object.__setattr__(self, "num_classes", num_classes)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "history_len", self.DEFAULT_HISTORY_LEN)
        object.__setattr__(self, "_len", 0)

        tensor_values: dict[str, torch.Tensor | None] = {}
        pending_defaults: list[str] = []
        length: int | None = None

        for name, spec in self._FIELD_REGISTRY.items():
            if name in kwargs:
                tensor = kwargs.pop(name)
                if tensor is not None:
                    tensor = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
                    tensor = tensor.to(device)
                    tensor_len = len(tensor)
                    if length is None:
                        length = tensor_len
                    elif tensor_len != length:
                        raise ValueError(f"Inconsistent tensor length in kwargs: expected {length}, got {tensor_len}")
                tensor_values[name] = tensor
            else:
                if spec.get("lazy", False):
                    tensor_values[name] = None
                else:
                    pending_defaults.append(name)

        if kwargs:
            raise AttributeError(
                f"{self.__class__.__name__} init does not support attributes: {', '.join(kwargs.keys())}"
            )

        object.__setattr__(self, "_len", length or 0)

        for name in pending_defaults:
            tensor_values[name] = self._get_alignment_properties(name)

        for name, tensor in tensor_values.items():
            object.__setattr__(self, name, tensor)

    def __len__(self) -> int:
        """Return the number of tracked instances."""
        return self._len

    def __getattribute__(self, name: str):
        tensor_attrs = object.__getattribute__(self, "_tensor_attrs")
        if name in tensor_attrs:
            attr_dict = object.__getattribute__(self, "__dict__")
            value = attr_dict.get(name, None)
            if value is None:
                lazy_attrs = object.__getattribute__(self, "_lazy_tensor_attrs")
                if name in lazy_attrs:
                    tensor = object.__getattribute__(self, "_get_alignment_properties")(name)
                    object.__setattr__(self, name, tensor)
                    return tensor
            return value
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        tensor_attrs = object.__getattribute__(self, "_tensor_attrs")
        if name in tensor_attrs:
            if value is not None:
                device = object.__getattribute__(self, "device")
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value, device=device)
                else:
                    value = value.to(device)
                expected_len = object.__getattribute__(self, "_len")
                if len(value) != expected_len:
                    raise ValueError(f"Tensor '{name}' length mismatch: expected {expected_len}, got {len(value)}")
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def __getitem__(self: T_Instances, item: Union[int, slice, torch.Tensor, List[int], List[bool]]) -> T_Instances:
        if isinstance(item, (list, tuple)):
            item = torch.tensor(item, device=self.device)

        if isinstance(item, int):
            if not (-len(self) <= item < len(self)):
                raise IndexError(f"Index {item} out of range for length {len(self)}")
            item = slice(item, item + 1)
        elif isinstance(item, torch.Tensor):
            if item.dim() != 1:
                raise ValueError(f"Index tensor must be 1D, got shape {item.shape}")
            if item.dtype == torch.bool:
                if item.numel() != len(self):
                    raise ValueError(f"Boolean index length mismatch: {item.numel()} vs {len(self)}")
            elif item.dtype not in (torch.int32, torch.int64, torch.long):
                raise TypeError(f"Unsupported tensor dtype: {item.dtype}")
            elif torch.any((item >= len(self)) | (item < -len(self))):
                raise IndexError("Tensor index out of range")
        elif not isinstance(item, slice):
            raise TypeError(f"Unsupported index type: {type(item)}")

        # Guard against silent field length drift before hitting device-side asserts.
        for name in self._tensor_attrs:
            tensor = object.__getattribute__(self, "__dict__").get(name, None)
            if tensor is not None and len(tensor) != len(self):
                raise ValueError(f"Tensor '{name}' length mismatch: {len(tensor)} vs expected {len(self)}")

        tensor_kwargs: Dict[str, Optional[torch.Tensor]] = {}
        attr_dict = object.__getattribute__(self, "__dict__")
        for name in self._tensor_attrs:
            tensor = attr_dict[name]
            # ===== DEBUG INDEXING BEGIN (delete after locating root cause) =====
            # Purpose:
            # - When CUDA throws a device-side assert, the reported stack is often delayed/asynchronous.
            # - This try/except captures the first failing field + index metadata so we can pinpoint
            #   whether the issue is a bad index, a device mismatch, or a specific tensor field.
            if tensor is None:
                tensor_kwargs[name] = None
                continue
            try:
                tensor_kwargs[name] = tensor[item]
            except Exception as e:
                item_desc = f"type={type(item)}"
                if isinstance(item, slice):
                    item_desc += f", slice=({item.start},{item.stop},{item.step})"
                elif isinstance(item, int):
                    item_desc += f", int={item}"
                elif isinstance(item, torch.Tensor):
                    item_desc += f", dtype={item.dtype}, device={item.device}, shape={tuple(item.shape)}"
                    # For integer indices, try to log min/max on CPU without touching the original CUDA tensor.
                    if item.dtype != torch.bool and item.numel() > 0:
                        try:
                            sample = item.flatten()[:2048].detach().to("cpu")
                            item_desc += f", sample_min={int(sample.min())}, sample_max={int(sample.max())}"
                        except Exception:
                            item_desc += ", sample_minmax=<unavailable>"
                    elif item.dtype == torch.bool:
                        try:
                            sample = item.flatten()[:8192].detach().to("cpu")
                            item_desc += f", sample_true={int(sample.sum())}/{int(sample.numel())}"
                        except Exception:
                            item_desc += ", sample_true=<unavailable>"

                tensor_desc = f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
                msg = (
                    "BaseInstances.__getitem__ failed during field slicing\n"
                    f"  field={name}\n"
                    f"  len(self)={len(self)}\n"
                    f"  tensor: {tensor_desc}\n"
                    f"  item: {item_desc}\n"
                    "Tip: run with CUDA_LAUNCH_BLOCKING=1 to get the true failing op location."
                )
                raise RuntimeError(msg) from e
            # ===== DEBUG INDEXING END =====

        return self.__class__(
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            frame=self.frame,
            device=self.device,
            **tensor_kwargs,
        )

    def _get_alignment_properties(self, name) -> torch.Tensor:
        """
        Generate default tensor with proper shape and dtype for tensor attributes.

        This method creates default tensors for tensor attributes when they are None
        or need to be initialized. The tensors are created with the specified length
        and appropriate dimensions based on the attribute type.

        Args:
            name (str): Name of the tensor attribute
            _len (int, optional): Length for the tensor. If None, uses self._len

        Returns:
            torch.Tensor: Default tensor with proper shape and dtype

        Raises:
            AttributeError: If the attribute name is not recognized

        Note:
            - query_embed, output_embed, last_output, long_memory: (len, hidden_dim)
            - ref_pts, boxes, last_appear_boxes: (len, 4) with center at (0.5, 0.5)
            - logits: (len, num_classes)
            - iou: (len,) scalar values
            - hist_boxes: (len, history_len, 4)
            - hist_frame_idx: (len, history_len) with -1 padding
            - hist_output_embed: (len, history_len, hidden_dim)
            - hist_ptr, hist_count: (len,)
            - ids, matched_idx, labels, disappear_time: (len,) with -1 padding
        """
        if name in self._tensor_attrs:
            f = self._FIELD_REGISTRY[name]
            res = torch.full(size=f["size"](self), fill_value=f["fill"], dtype=f["dtype"], device=self.device)
            if name in ["ref_pts", "boxes", "last_appear_boxes"]:
                res[:, 2:] = 0.0
            return res
        else:
            raise AttributeError(f"get_alignment_properties get an unknown attribute name:{name}")

    # ================== External API ================== #
    @property
    def area(self) -> torch.Tensor:
        if self._len == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)
        return self.boxes[:, 2] * self.boxes[:, 3]

    @property
    def scores(self) -> torch.Tensor:
        if self._len == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.device)
        return torch.max(logits_to_scores(self.logits), dim=1).values

    def clone(self: T_Instances) -> T_Instances:
        attr_dict = object.__getattribute__(self, "__dict__")
        tensor_kwargs = {
            name: (attr_dict[name].clone() if attr_dict[name] is not None else None) for name in self._tensor_attrs
        }
        return self.__class__(
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            frame=self.frame,
            device=self.device,
            **tensor_kwargs,
        )

    @classmethod
    def init_tracks(
        cls: type[T_Instances],
        batch: MutableMapping[str, Any],
        hidden_dim: int,
        num_classes: int,
        device: torch.device = torch.device("cpu"),
        kwargs: Optional[List[Dict]] = None,
    ) -> List[T_Instances]:
        if "infos" not in batch:
            raise KeyError("batch must contain 'infos' to initialize track instances")

        infos = batch["infos"]
        if len(infos) == 0:
            return []

        batch_size = len(infos[0])
        if kwargs is None:
            kwargs = [{} for _ in range(batch_size)]
        elif len(kwargs) != batch_size:
            raise ValueError(f"kwargs length {len(kwargs)} does not match batch size {batch_size}")

        instances: List[T_Instances] = []
        for idx in range(batch_size):
            inst_kwargs = kwargs[idx] if kwargs[idx] is not None else {}
            instances.append(
                cls(
                    frame_height=infos[0][idx]["org_shape"][1],
                    frame_width=infos[0][idx]["org_shape"][0],
                    hidden_dim=hidden_dim,
                    num_classes=num_classes,
                    frame=0,
                    device=device,
                    **inst_kwargs,
                )
            )
        return instances

    @classmethod
    def cat(cls: type[T_Instances], instance1: T_Instances, instance2: T_Instances) -> T_Instances:
        if instance1 is None or instance2 is None or type(instance1) is not type(instance2):
            raise TypeError("Both instances must be non-null and of the same type for concatenation")
        if len(instance1) == 0:
            return instance2.clone()
        if len(instance2) == 0:
            return instance1.clone()

        tensor_kwargs: Dict[str, Optional[torch.Tensor]] = {}
        attr_dict_1 = object.__getattribute__(instance1, "__dict__")
        attr_dict_2 = object.__getattribute__(instance2, "__dict__")
        for name in cls._tensor_attrs:
            t1 = attr_dict_1[name]
            t2 = attr_dict_2[name]
            spec = cls._FIELD_REGISTRY[name]
            if spec.get("lazy", False):
                if t1 is None and t2 is None:
                    tensor_kwargs[name] = None
                    continue
                lhs = t1 if t1 is not None else instance1._get_alignment_properties(name)
                rhs = t2 if t2 is not None else instance2._get_alignment_properties(name)
                tensor_kwargs[name] = torch.cat((lhs, rhs), dim=0)
            else:
                tensor_kwargs[name] = torch.cat((t1, t2), dim=0)

        return cls(
            frame_height=instance1.frame_height,
            frame_width=instance1.frame_width,
            hidden_dim=instance1.hidden_dim,
            num_classes=instance1.num_classes,
            frame=instance1.frame,
            device=instance1.device,
            **tensor_kwargs,
        )

    @classmethod
    def as_like(cls: type[T_Instances], instance: "BaseInstances", **kwargs: Any) -> T_Instances:
        return cls(
            frame_height=instance.frame_height,
            frame_width=instance.frame_width,
            hidden_dim=instance.hidden_dim,
            num_classes=instance.num_classes,
            frame=instance.frame,
            device=instance.device,
            **kwargs,
        )
