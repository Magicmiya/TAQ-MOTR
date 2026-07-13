import os
from typing import Any

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ..core import BaseVisualTask


def _is_main_process() -> bool:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return True
    return torch.distributed.get_rank() == 0


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class GradMonitorTask(BaseVisualTask):
    def __init__(self, task_name: str, cfg: dict, mode: str, root_dir: str):
        super().__init__(task_name, cfg, mode, root_dir)
        self.interval = int(cfg.get("interval", 100))
        self.enabled_epochs = {int(x) for x in cfg.get("enabled_epochs", [])}
        self.modules = self._normalize_modules(cfg=cfg)
        self.log_dir_name = str(cfg.get("log_dir_name", "grad_log"))
        self.writer: SummaryWriter | None = None
        self._pending_stats: dict[str, dict[str, float]] = {}
        self._pending_step: int | None = None

    def init(self):
        if not self.enabled or not _is_main_process():
            return
        log_dir = os.path.join(self.root_dir, self.log_dir_name)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def before_grad_clip(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
    ):
        if not self._should_log(epoch=epoch, global_step=global_step):
            self._pending_stats = {}
            self._pending_step = None
            return
        self._pending_stats = self._collect_grad_stats(model=model, optimizer=optimizer)
        self._pending_step = int(global_step)

    def after_grad_clip(
        self,
        pre_clip_norm: torch.Tensor | float | None,
        max_norm: float,
        global_step: int,
    ):
        if self.writer is None or self._pending_step != int(global_step):
            return
        pre_clip_value = self._to_float(pre_clip_norm)
        clip_coef = min(float(max_norm) / (pre_clip_value + 1e-6), 1.0) if max_norm > 0 else 1.0

        for module_name, stats in self._pending_stats.items():
            for stat_name, value in stats.items():
                self.writer.add_scalar(f"{module_name}/{stat_name}", value, global_step=global_step)
            self.writer.add_scalar(
                f"{module_name}/post_clip_update_norm",
                stats["update_norm"] * clip_coef,
                global_step=global_step,
            )
            self.writer.add_scalar(
                f"{module_name}/post_clip_update_ratio",
                stats["update_ratio"] * clip_coef,
                global_step=global_step,
            )

        self.writer.add_scalar("global/pre_clip_norm", pre_clip_value, global_step=global_step)
        self.writer.add_scalar("global/clip_coef", clip_coef, global_step=global_step)
        self.writer.add_scalar("global/max_norm", float(max_norm), global_step=global_step)

    def compute_global_grad_norm(self, model: nn.Module) -> torch.Tensor | None:
        if not self.enabled:
            return None
        model_without_ddp = _unwrap_model(model)
        device = next(model_without_ddp.parameters()).device
        grad_sq = torch.zeros((), device=device)
        for param in model_without_ddp.parameters():
            if param.grad is None:
                continue
            grad_data = param.grad.detach().float()
            grad_sq = grad_sq + grad_data.square().sum()
        return torch.sqrt(grad_sq)

    def close(self):
        if self.writer is None:
            return
        self.writer.flush()
        self.writer.close()
        self.writer = None

    def _should_log(self, epoch: int, global_step: int) -> bool:
        if not self.enabled or self.writer is None:
            return False
        if self.interval <= 0 or int(global_step) % self.interval != 0:
            return False
        if self.enabled_epochs and int(epoch) not in self.enabled_epochs:
            return False
        return True

    @staticmethod
    def _normalize_modules(cfg: dict[str, Any]) -> dict[str, list[str]]:
        modules_cfg = cfg.get("modules", {})
        if not isinstance(modules_cfg, dict):
            raise TypeError("Visualizer.tasks.grad_monitor.modules must be {module_name: [param_prefix, ...]}.")

        modules: dict[str, list[str]] = {}
        for module_name, prefixes in modules_cfg.items():
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            if not isinstance(prefixes, list) or not all(isinstance(prefix, str) for prefix in prefixes):
                raise TypeError(f"Visualizer.tasks.grad_monitor.modules.{module_name} must be a string list.")
            modules[str(module_name)] = [prefix for prefix in prefixes if prefix]
        return modules

    def _collect_grad_stats(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict[str, float]]:
        if not self.modules:
            return {}

        model_without_ddp = _unwrap_model(model)
        device = next(model_without_ddp.parameters()).device
        module_tensors = {
            module_name: {
                "grad_sq": torch.zeros((), device=device),
                "param_sq": torch.zeros((), device=device),
                "update_sq": torch.zeros((), device=device),
                "grad_max": torch.zeros((), device=device),
            }
            for module_name in self.modules
        }
        param_lrs = {
            id(param): float(group.get("lr", 0.0))
            for group in optimizer.param_groups
            for param in group.get("params", [])
        }

        # Read existing gradients only; no hook, graph retention, or gradient copy is introduced.
        for param_name, param in model_without_ddp.named_parameters():
            matched_modules = [
                module_name
                for module_name, prefixes in self.modules.items()
                if any(param_name.startswith(prefix) for prefix in prefixes)
            ]
            if not matched_modules or not param.requires_grad:
                continue

            param_data = param.detach().float()
            param_sq = param_data.square().sum()
            grad = param.grad.detach() if param.grad is not None else None
            grad_sq = torch.zeros((), device=device)
            update_sq = torch.zeros((), device=device)
            grad_max = torch.zeros((), device=device)
            if grad is not None:
                grad_data = grad.float()
                grad_sq = grad_data.square().sum()
                lr = param_lrs.get(id(param), 0.0)
                update_sq = grad_sq * (lr**2)
                grad_max = grad_data.abs().max()

            for module_name in matched_modules:
                tensors = module_tensors[module_name]
                tensors["param_sq"] = tensors["param_sq"] + param_sq
                tensors["grad_sq"] = tensors["grad_sq"] + grad_sq
                tensors["update_sq"] = tensors["update_sq"] + update_sq
                tensors["grad_max"] = torch.maximum(tensors["grad_max"], grad_max)

        grad_stats: dict[str, dict[str, float]] = {}
        for module_name, tensors in module_tensors.items():
            grad_norm = float(torch.sqrt(tensors["grad_sq"]).item())
            param_norm = float(torch.sqrt(tensors["param_sq"]).item())
            update_norm = float(torch.sqrt(tensors["update_sq"]).item())
            grad_stats[module_name] = {
                "grad_norm": grad_norm,
                "param_norm": param_norm,
                "update_norm": update_norm,
                "update_ratio": update_norm / (param_norm + 1e-12),
                "grad_max": float(tensors["grad_max"].item()),
            }
        return grad_stats

    @staticmethod
    def _to_float(value: torch.Tensor | float | None) -> float:
        if value is None:
            return 0.0
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().item())
        return float(value)
