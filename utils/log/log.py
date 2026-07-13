# @Author       : Ruopeng Gao
# @Date         : 2022/7/13
# @Description  :
import torch

from typing import Any
from collections import deque, defaultdict
from utils.utils import is_dist, dist_world_size

DEFAULT_TB_LOSS_LOG_LEVEL = 1
TB_LOSS_LOG_LEVELS = {
    0: {
        "name": "minimal",
        "views": ("overview", "label_stats"),
    },
    1: {
        "name": "compact",
        "views": ("overview", "by_type", "branch_summary", "frame_overview", "label_stats"),
    },
    2: {
        "name": "detailed",
        "views": (
            "overview",
            "by_type",
            "branch_summary",
            "frame_overview",
            "frame_by_type",
            "component_summary",
            "label_stats",
        ),
    },
    3: {
        "name": "diagnostic",
        "views": (
            "overview",
            "by_type",
            "branch_summary",
            "frame_overview",
            "frame_by_type",
            "component_summary",
            "frame_branch",
            "frame_component",
            "label_stats",
        ),
    },
}
TB_LOSS_LOG_LEVEL_ALIASES = {
    "minimal": 0,
    "compact": 1,
    "default": 1,
    "detailed": 2,
    "diagnostic": 3,
    "debug": 3,
}


class Value:
    def __init__(self, window_size: int = 100):
        self.value_deque = deque(maxlen=window_size)
        self.total_value = 0.0
        self.total_count = 0

        self.value_sync: None | torch.Tensor = None
        self.total_value_sync = None
        self.total_count_sync = None

    def update(self, value):
        self.value_deque.append(value)
        self.total_value += value
        self.total_count += 1

    def sync(self):
        if is_dist():
            torch.distributed.barrier()
            value_list_gather = [None] * dist_world_size()
            value_count_gather = [None] * dist_world_size()
            torch.distributed.all_gather_object(value_list_gather, list(self.value_deque))
            torch.distributed.all_gather_object(value_count_gather, [self.total_value, self.total_count])
            value_list = [v for v_list in value_list_gather for v in v_list]
            self.value_sync = torch.as_tensor(value_list)
            self.total_value_sync = sum([_[0] for _ in value_count_gather])
            self.total_count_sync = int(sum([_[1] for _ in value_count_gather]))
        else:
            self.value_sync = torch.as_tensor(list(self.value_deque))
            self.total_value_sync = self.total_value
            self.total_count_sync = self.total_count
        return

    @property
    def avg(self):
        self.check_sync()
        return self.value_sync.mean().item()

    @property
    def global_avg(self):
        self.check_sync()
        return self.total_value_sync / self.total_count_sync

    def check_sync(self):
        if self.value_sync is None:
            raise RuntimeError(f"Be sure to use .sync() before metric statistic.")
        return


class MetricLog:
    def __init__(self, tb_loss_log_level: int | str | None = None):
        self._metrics = defaultdict(Value)
        self._scalars: dict[str, Value] = {}
        self._tb_data: defaultdict[str, dict] = defaultdict(dict)
        self._tb_loss_log_level = self._normalize_tb_loss_log_level(tb_loss_log_level)
        self._tb_loss_views = set(TB_LOSS_LOG_LEVELS[self._tb_loss_log_level]["views"])
        self._pending: dict[str, float] = defaultdict(float)
        self._metric_to_tag: dict[str, tuple[str, str]] = {}

    def update_detail(self, log_dict: dict):
        tb_scalars = log_dict.pop("__tb_scalars__", None)
        for key, value in log_dict.items():
            if not isinstance(key, int):
                continue
            self._update_frame_losses(key, value)
        self._update_tb_scalars(tb_scalars)
        self._flush_pending()

    def sync(self):
        for name, value in self._metrics.items():
            value.sync()
        return

    def _get(self, vals, mode: str):
        """
        Recursively traverse nested dictionary and sum all Value objects.
        Args:
            vals: Can be a Value object, dict, or other type
            mode (str): 'iters' for .avg, otherwise for .global_avg
        Returns:
            float: Sum of all Value objects in the nested structure
        Raises:
            TypeError: When encountering unexpected data types
        """
        if isinstance(vals, Value):
            return vals.avg if mode == "iters" else vals.global_avg
        elif isinstance(vals, dict):
            total = 0.0
            for v in vals.values():
                total += self._get(v, mode)
            return total
        else:
            raise TypeError(
                f"Unexpected data type in _get method: {type(vals)}. "
                f"Expected Value or dict, but got {type(vals)} with value: {vals}. "
                f"Mode: {mode}. "
                f"This usually indicates a data structure mismatch in MetricLog."
            )

    def get_tb(self, mode: str = "iter", level=2):
        tags = []
        tag_scalar_dicts = []
        for key, value in self._tb_data.items():
            if not value:
                continue
            tags.append(key)
            scalar = {}
            for k, v in value.items():
                if isinstance(v, Value):
                    scalar[k] = v.avg if mode == "iters" else v.global_avg
                else:
                    scalar[k] = self._get(value[k], mode)
            tag_scalar_dicts.append(scalar)
        return tags, tag_scalar_dicts

    def __str__(self):
        total_view = self._tb_data.get("train/loss_overview", {})
        parts = []
        for name, value in total_view.items():
            if '_' in name:
                continue
            parts.append(f"{name} = {value.avg:.4f} ({value.global_avg:.4f})")
        return "; ".join(parts)

    # ===================== frame log helpers =====================
    def __getitem__(self, item):
        return self._scalars[item]

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            raise TypeError(
                "MetricLog.__setitem__ only accepts scalar values. "
                "Use MetricLog.update_detail(log_dict) for structured inputs."
            )
        value = self._to_float(value)
        metric = self._metrics[str(key)]
        metric.update(value)
        self._scalars[str(key)] = metric
        if key == "total_loss":
            self._tb_data["train/loss_overview"]["total"] = metric
        return

    def _update_frame_losses(self, frame_idx: Any, loss_dict: dict):
        assert isinstance(frame_idx, int), TypeError(
            f"frame_idx must be int put get {type(frame_idx)} with value: {frame_idx}")
        frame_key = f'F{frame_idx}'
        for loss_name, group_vals in loss_dict.items():
            alias = self._normalize_loss_alias(loss_name)
            for group_name, raw_val in group_vals.items():
                scalar = self._to_float(raw_val)
                branch, component = self._categorize_group(group_name)
                self._record_loss_views(
                    frame_key=frame_key,
                    alias=alias,
                    branch=branch,
                    component=component,
                    value=scalar,
                )
        return

    def _record_loss_views(self, frame_key: str, alias: str, branch: str, component: str, value: float):
        """Record TensorBoard loss views selected by the configured detail level."""
        # TensorBoard loss tags use the new branch naming only. Historical
        # group names such as deN/DN are parsed as sources but never emitted as tags.
        if "overview" in self._tb_loss_views:
            self._update_tag_metric("train/loss_overview", branch, value)
        if "by_type" in self._tb_loss_views:
            self._update_tag_metric("train/loss_by_type", alias, value)
        if "branch_summary" in self._tb_loss_views:
            self._update_tag_metric(f"train/loss_{branch}", alias, value)
        if "frame_overview" in self._tb_loss_views:
            self._update_tag_metric("train/frame_loss_overview", f"{frame_key}_total", value)
        if "frame_by_type" in self._tb_loss_views:
            self._update_tag_metric("train/frame_loss_by_type", f"{frame_key}_{alias}", value)
        if "component_summary" in self._tb_loss_views:
            self._update_tag_metric(f"train/loss_{branch}_components", f"{component}_{alias}", value)
        if "frame_branch" in self._tb_loss_views:
            self._update_tag_metric("train/frame_loss_by_branch", f"{frame_key}_{branch}", value)
        if "frame_component" in self._tb_loss_views:
            self._update_tag_metric("train/frame_loss_by_component", f"{frame_key}_{component}", value)
        return

    def _update_tag_metric(self, tag: str, scalar_key: str, value: float):
        metric_name = f"{tag}_{scalar_key}"
        self._pending[metric_name] += value
        self._metric_to_tag[metric_name] = (tag, scalar_key)
        return

    def _update_tb_scalars(self, tb_scalars: Any):
        if not isinstance(tb_scalars, dict):
            return
        for tag, scalars in tb_scalars.items():
            if not isinstance(scalars, dict):
                continue
            if str(tag) == "label_stats" and "label_stats" in self._tb_loss_views:
                self._update_label_stats(scalars)
                continue
            for name, val in scalars.items():
                self.add_scalar(tag=str(tag), name=str(name), value=val)
        return

    def _update_label_stats(self, scalars: dict):
        for name, val in scalars.items():
            stat_name = self._normalize_label_stat_name(str(name))
            if stat_name.endswith("_gt_total"):
                self.add_scalar("train/label_stats_gt", stat_name, val)
            else:
                self.add_scalar("train/label_stats_supervision", stat_name, val)
        return

    @classmethod
    def _normalize_tb_loss_log_level(cls, level: int | str | None) -> int:
        if level is None:
            return DEFAULT_TB_LOSS_LOG_LEVEL
        if isinstance(level, str):
            key = level.strip().lower()
            if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
                level = int(key)
            elif key in TB_LOSS_LOG_LEVEL_ALIASES:
                level = TB_LOSS_LOG_LEVEL_ALIASES[key]
            else:
                valid = sorted({*TB_LOSS_LOG_LEVEL_ALIASES.keys(), *[str(k) for k in TB_LOSS_LOG_LEVELS]})
                raise ValueError(f"Unknown tb_loss_log_level={level!r}. Valid levels: {valid}")
        if int(level) not in TB_LOSS_LOG_LEVELS:
            valid = sorted(TB_LOSS_LOG_LEVELS.keys())
            raise ValueError(f"Unknown tb_loss_log_level={level!r}. Valid levels: {valid}")
        return int(level)

    @staticmethod
    def _normalize_loss_alias(loss_name: str) -> str:
        alias = str(loss_name).split("_")[-1].lower()
        if alias == "l1":
            return "l1"
        return alias

    @staticmethod
    def _normalize_label_stat_name(name: str) -> str:
        return name.replace("_dn_", "_det_only_").replace("_dn", "_det_only")

    @staticmethod
    def _categorize_group(group_name: str) -> tuple[str, str]:
        lower = group_name.lower()
        if lower == "main_dense":
            return "qpn_branch", "qpn_main"
        if lower.startswith("main"):
            return "main_branch", "main"
        if lower.startswith("aux"):
            layer = group_name.split("_")[-1]
            return "aux_branch", f"aux_{layer}"
        if lower.startswith("qpn_aux"):
            layer = group_name.rsplit("_", 1)[-1]
            return "aux_branch", f"qpn_aux_{layer}" if layer.isdigit() else "qpn_aux"
        if lower.startswith("qpn_dn"):
            return "det_only_branch", "qpn_det_only"
        if lower.startswith("dec"):
            sub_name = group_name.split("_")[-1]
            return "decoder_branch", "dec_" + sub_name
        if lower.startswith("det_dn_outputs_gt"):
            return "det_only_branch", "det_only_gt"
        if lower.startswith("det_dn_outputs_"):
            layer = group_name.rsplit("_", 1)[-1]
            if layer.isdigit():
                return "det_only_branch", f"det_only_{layer}"
        if lower.startswith("det_dn"):
            return "det_only_branch", "det_only"
        if lower.startswith("dn"):
            return "det_only_branch", "det_only"
        return "other_branch", "other"

    def _flush_pending(self):
        if not self._pending:
            return
        for metric_name, total in self._pending.items():
            metric = self._metrics[metric_name]
            metric.update(total)
            tag, scalar_key = self._metric_to_tag[metric_name]
            self._tb_data[tag][scalar_key] = metric
        self._pending.clear()
        return

    def add_scalar(self, tag: str, name: str, value: Any):
        """Record a scalar metric under a custom TensorBoard tag."""
        value = self._to_float(value)
        self._update_tag_metric(tag, str(name), value)
        self._flush_pending()
        return

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)


def merge_dicts(dicts: list[dict]) -> dict:
    merged = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in merged.keys():
                merged[k] = list()
            merged[k] += v
    return merged
