import os
import select
import sys
from copy import deepcopy
from typing import Any, cast

import torch
import torch.nn as nn
from torch import distributed as dist
import numpy as np
import random


def dist_rank():
    if not is_dist():
        return 0
    else:
        return dist.get_rank()


def dist_world_size():
    if is_dist():
        return dist.get_world_size()
    else:
        return 1


def is_dist():
    if not (dist.is_available() and dist.is_initialized()):
        return False
    return True


def is_main_process():
    return dist_rank() == 0


def set_seed(seed: int):
    seed = seed + dist_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # If you don't want to wait until the universe is silent, do not use this below code :)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return


def show_params_name(model: torch.nn.Module, idx: list):
    print("\n" + "=" * 80)
    print("MODEL PARAMETER ANALYSIS")
    print("=" * 80)

    # Get all parameters with their names and indices
    param_info = []
    for name, param in model.named_parameters():
        param_info.append((name, param.numel(), param.shape))

    # Print total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Total named parameters: {len(param_info)}")

    # Check specific parameter indices mentioned in the error
    error_indices = idx

    print(f"\nAnalyzing error parameter indices: {error_indices}")
    print("-" * 80)

    # Calculate cumulative parameter count to find which parameters correspond to these indices
    cumsum = 0
    for i, (name, numel, shape) in enumerate(param_info):
        if i in error_indices or any(idx >= cumsum and idx < cumsum + numel for idx in error_indices):
            print(f"Parameter {i:3d}: {name:<50} | Shape: {str(shape):<20} | Elements: {numel:>8}")

            # Check if any error indices fall within this parameter's range
            param_start = cumsum
            param_end = cumsum + numel - 1
            matching_indices = [idx for idx in error_indices if param_start <= idx <= param_end]
            if matching_indices:
                print(f"  -> Contains error indices: {matching_indices}")
                print(f"  -> Parameter range: {param_start} to {param_end}")

        cumsum += numel

    print("=" * 80)
    print("END PARAMETER ANALYSIS")
    print("=" * 80 + "\n")

    # Additional analysis for query_updater parameters specifically
    print("\n" + "=" * 80)
    print("QUERY_UPDATER PARAMETER ANALYSIS")
    print("=" * 80)

    query_updater_params = []
    for name, param in model.named_parameters():
        if "query_updater" in name:
            query_updater_params.append((name, param.numel(), param.shape))

    print(f"Query updater parameters count: {len(query_updater_params)}")
    print(f"Query updater total elements: {sum(p[1] for p in query_updater_params)}")

    # Calculate starting index for query_updater parameters
    cumsum_before_query = 0
    for name, param in model.named_parameters():
        if "query_updater" not in name:
            cumsum_before_query += param.numel()
        else:
            break

    print(f"Query updater parameters start at index: {cumsum_before_query}")
    print("-" * 80)

    cumsum_query = cumsum_before_query
    for i, (name, numel, shape) in enumerate(query_updater_params):
        param_start = cumsum_query
        param_end = cumsum_query + numel - 1
        print(f"Query Param {i:2d}: {name:<60} | Shape: {str(shape):<15} | Range: {param_start:>6} to {param_end:>6}")

        # Check if any error indices fall within this parameter's range
        matching_indices = [idx for idx in error_indices if param_start <= idx <= param_end]
        if matching_indices:
            print(f"  *** CONTAINS ERROR INDICES: {matching_indices} ***")

        cumsum_query += numel

    print("=" * 80)
    print("END QUERY_UPDATER ANALYSIS")
    print("=" * 80 + "\n")


def _stage_value(value: Any, default: Any, stage_idx: int | None = None) -> Any:
    if value is None:
        return default
    if not isinstance(value, list):
        return value
    if not value:
        return default
    return value[0 if stage_idx is None else min(stage_idx, len(value) - 1)]


def _stage_steps(value: Any) -> list[int]:
    steps = [int(x) for x in (value or [])]
    return steps[1:] if steps[:1] == [0] else steps


def _build_stage_defaults(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cast(dict[str, Any], config.get("Dataset", {}))
    decoder_cfg = cast(dict[str, Any], config.get("Decoder", {}))
    criterion_cfg = cast(dict[str, Any], config.get("Criterion", {}))
    lcm_cfg = cast(dict[str, Any], config.get("Life_cycle_management", {}))
    get = _stage_value
    return {
        "sample_length": int(get(dataset_cfg.get("sample_length", dataset_cfg.get("sample_lengths")), 2)),
        "batch_size": int(get(dataset_cfg.get("batch_size"), 1)),
        "sample_mode": str(get(dataset_cfg.get("sample_mode", dataset_cfg.get("sample_modes")), "fixed_interval")),
        "sample_interval": int(get(dataset_cfg.get("sample_interval", dataset_cfg.get("sample_intervals")), 1)),
        "use_aux_loss": bool(criterion_cfg.get("aux_loss", True)),
        "use_det_dn_aux": True,
        "use_dn": bool(int(decoder_cfg.get("num_denoising", 0)) > 0),
        "high_conf_threshold": float(lcm_cfg.get("high_conf_threshold", 0.5)),
    }


def build_stage_policy(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = cast(dict[str, Any], config["Training"])
    dataset_cfg = cast(dict[str, Any], config.get("Dataset", {}))
    stage_cfg = deepcopy(cast(dict[str, Any], train_cfg.get("stage_policy", {})))
    defaults = _build_stage_defaults(config)

    if "stages" not in stage_cfg:
        sample_steps = _stage_steps(stage_cfg.get("sample_steps", dataset_cfg.get("sample_steps", [])))
        drop_det_aux = int(stage_cfg.get("det_dn_aux_drop_epoch", 100000))
        stop_dn = int(stage_cfg.get("stop_dn_epoch", train_cfg.get("stop_DN", 100000)))
        final_only = int(stage_cfg.get("final_only_epoch", 100000))
        control_steps = sorted({*sample_steps, *[x for x in (drop_det_aux, stop_dn, final_only) if x < 100000]})
        sample_fields = (
            ("sample_length", "sample_lengths", int),
            ("batch_size", "batch_size", int),
            ("sample_mode", "sample_modes", str),
            ("sample_interval", "sample_intervals", int),
        )
        stages = []
        for stage_idx in range(len(control_steps) + 1):
            start_epoch = 0 if stage_idx == 0 else control_steps[stage_idx - 1]
            sample_stage = sum(start_epoch >= step for step in sample_steps)
            stage = deepcopy(defaults)
            stage.update(
                {
                    key: convert(_stage_value(dataset_cfg.get(cfg_key), defaults[key], sample_stage))
                    for key, cfg_key, convert in sample_fields
                }
            )
            stage.update(
                {
                    "use_det_dn_aux": start_epoch < drop_det_aux,
                    "use_dn": start_epoch < stop_dn,
                    "use_aux_loss": start_epoch < final_only,
                }
            )
            stages.append(stage)
        return {"sample_steps": control_steps, "stages": stages}

    sample_steps = _stage_steps(stage_cfg.get("sample_steps", []))
    stages = []
    current = deepcopy(defaults)
    for raw_stage in stage_cfg.get("stages", []):
        current = {**current, **(raw_stage or {})}
        stages.append(deepcopy(current))
    if not stages:
        stages = [deepcopy(defaults)]
    if len(stages) != len(sample_steps) + 1:
        raise ValueError(
            f"stage_policy.stages length {len(stages)} must equal len(sample_steps)+1 = {len(sample_steps) + 1}"
        )
    return {"sample_steps": sample_steps, "stages": stages}


def resolve_stage(stage_policy: dict[str, Any], epoch: int) -> tuple[int, dict[str, Any]]:
    sample_steps = cast(list[int], stage_policy.get("sample_steps", []))
    stages = cast(list[dict[str, Any]], stage_policy.get("stages", []))
    if not stages:
        raise RuntimeError("Resolved empty stage list.")
    stage_idx = min(sum(epoch >= int(step) for step in sample_steps), len(stages) - 1)
    return stage_idx, deepcopy(stages[stage_idx])


def apply_stage_lr(optimizer: torch.optim.Optimizer, lr_names: list[str], stage_cfg: dict[str, Any]):
    if "lr" not in stage_cfg:
        return
    stage_lr = stage_cfg["lr"]
    if not isinstance(stage_lr, (list, tuple)):
        raise TypeError(f"stage lr must be a list/tuple ordered as {lr_names}, got {type(stage_lr).__name__}.")
    if len(stage_lr) != len(optimizer.param_groups):
        raise ValueError(
            f"stage lr length {len(stage_lr)} must match optimizer param groups {len(optimizer.param_groups)} "
            f"ordered as {lr_names}."
        )
    for name, lr in zip(lr_names, stage_lr):
        if not isinstance(lr, (int, float)):
            raise TypeError(f"stage lr for '{name}' must be numeric, got {type(lr).__name__}.")
        if float(lr) < 0.0:
            raise ValueError(f"stage lr for '{name}' must be non-negative, got {lr}.")

    # Stage lr explicitly controls each optimizer param group; no global scheduler is applied.
    for group, lr in zip(optimizer.param_groups, stage_lr):
        group["lr"] = float(lr)


def _remove_old_best_checkpoint(output_dir: str, metric_name: str):
    prefix = f"checkpoint_best_{metric_name}_epoch_"
    for file_name in os.listdir(output_dir):
        if file_name.startswith(prefix) and file_name.endswith(".pth"):
            os.remove(os.path.join(output_dir, file_name))


def update_best_checkpoints(
    config: dict,
    logger: Any,
    train_states: dict,
    summary: dict | None,
    epoch: int,
):
    if not is_main_process() or not summary:
        return

    output_dir = config["OUTPUTS_DIR"]
    last_ckpt_path = os.path.join(output_dir, "checkpoint_last.pth")
    if not os.path.isfile(last_ckpt_path):
        return

    best_metrics = train_states.setdefault("best_eval_metrics", {"HOTA": float("-inf"), "MOTA": float("-inf")})
    best_epochs = train_states.setdefault("best_eval_epochs", {"HOTA": -1, "MOTA": -1})
    for metric_name in ("HOTA", "MOTA"):
        if metric_name not in summary:
            continue
        metric_value = float(summary[metric_name])
        if metric_value <= float(best_metrics.get(metric_name, float("-inf"))):
            continue
        _remove_old_best_checkpoint(output_dir=output_dir, metric_name=metric_name)
        dst_name = f"checkpoint_best_{metric_name}_epoch_{epoch}.pth"
        from module import copy_checkpoint

        copy_checkpoint(
            root_dir=output_dir,
            src_name="checkpoint_last.pth",
            dst_name=dst_name,
            logger=logger,
        )
        best_metrics[metric_name] = metric_value
        best_epochs[metric_name] = int(epoch)
        logger.show(
            head="[Best] ",
            log=f"{metric_name}={metric_value:.4f} at epoch={epoch}, saved {dst_name}",
            write=True,
        )


def get_param_groups(config: dict, model: nn.Module) -> tuple[list[dict], list[str]]:
    """
    Configure different learning rates for different model components.
    """

    def match_keywords(_name: str, _keywords: list[str]):
        for _keyword in _keywords:
            if _keyword in _name:
                return True
        return False

    param_groups = [{"params": [], "lr": config["lr_rate"][key]} for key in config["lr_rate"]]
    lr_keys = list(config["lr_rate"].keys())
    for n, p in model.named_parameters():
        matched = False
        for i, key in enumerate(lr_keys[:-1]):
            if match_keywords(n, [f"{key}."]) and p.requires_grad:
                param_groups[i]["params"].append(p)
                matched = True
                break
        if not matched and p.requires_grad:
            param_groups[-1]["params"].append(p)
    return param_groups, lr_keys


def sync_manual_stop(local_stop: bool, device: torch.device) -> bool:
    """
    Synchronize the manual-stop signal across distributed ranks.
    """
    if not is_dist():
        return local_stop
    stop_flag = torch.tensor([1 if local_stop else 0], dtype=torch.int32, device=device)
    dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
    return bool(stop_flag.item())


def start_manual_save_listener(logger: Any, enabled: bool = True) -> tuple[dict, str]:
    """
    Build a manual save controller for interactive "save + Enter".
    """
    controller = {
        "active": False,
        "stdin": None,
        "command": "save",
        "stop_requested": False,
    }
    if not enabled:
        return controller, "save"

    save_cmd = str(controller["command"])
    if not is_main_process():
        return controller, save_cmd

    stdin = getattr(sys, "stdin", None)
    if stdin is None or not stdin.isatty():
        logger.show(
            head="[Control] ",
            log="interactive save is disabled because stdin is not an interactive terminal.",
        )
        return controller, save_cmd

    controller["active"] = True
    controller["stdin"] = stdin
    logger.show(head="[Control] ", log=f"type '{save_cmd}' + Enter to save checkpoint and exit.")
    return controller, save_cmd


def poll_manual_save_command(controller: dict) -> bool:
    """
    Poll stdin without blocking; return True once a save command is received.
    """
    if not bool(controller.get("active", False)):
        return bool(controller.get("stop_requested", False))
    if bool(controller.get("stop_requested", False)):
        return True

    stdin = controller.get("stdin", None)
    if stdin is None:
        controller["active"] = False
        return False

    try:
        readable, _, _ = select.select([stdin], [], [], 0.0)
    except (OSError, ValueError):
        controller["active"] = False
        return False

    if not readable:
        return False

    cmd = stdin.readline()
    if cmd == "":
        controller["active"] = False
        return False
    if cmd.strip().lower() == str(controller.get("command", "save")):
        controller["stop_requested"] = True
        return True
    return False
