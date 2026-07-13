# @Author       :
# @Date         :
# @Description  : NestedTensor，modified from Me-MOTR
# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)
import os
import copy
import math
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor

import yaml

import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim

from utils import is_dist, is_main_process, Logger
from typing import Any


_CHECKPOINT_COPY_EXECUTOR = None
_CHECKPOINT_COPY_FUTURES = []


def get_model(model) -> Any:
    return model if is_dist() is False else model.module


def save_checkpoint(
    model: nn.Module,
    path: str,
    states: dict | None = None,
    optimizer: optim.Optimizer | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    backup: bool = True,
):
    model = get_model(model)
    if is_main_process():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if backup and os.path.exists(path):
            backup_path = path + ".backup"
            shutil.move(path, backup_path)

        save_state = {
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            'states': states,
        }
        temp_path = path + ".tmp"
        torch.save(save_state, temp_path)
        os.rename(temp_path, path)
    else:
        pass
    return


def _checkpoint_path(root_dir: str, name: str) -> str:
    return name if os.path.isabs(name) else os.path.join(root_dir, name)


def _get_checkpoint_copy_executor():
    global _CHECKPOINT_COPY_EXECUTOR
    if _CHECKPOINT_COPY_EXECUTOR is None:
        _CHECKPOINT_COPY_EXECUTOR = ThreadPoolExecutor(max_workers=1)
    return _CHECKPOINT_COPY_EXECUTOR


def _copy_checkpoint_snapshot(snapshot_path: str, tmp_path: str, dst_path: str):
    try:
        shutil.copyfile(snapshot_path, tmp_path)
        os.replace(tmp_path, dst_path)
    finally:
        for path in (snapshot_path, tmp_path):
            if os.path.exists(path):
                os.remove(path)


def copy_checkpoint(
    root_dir: str | None = None,
    dst_name: str | None = None,
    src_name: str = "checkpoint_last.pth",
    logger: Logger | None = None,
    non_blocking: bool = True,
    wait: bool = False,
):
    """
    Copy a derived checkpoint from checkpoint_last.pth by default.

    Set wait=True to join pending non-blocking copies; if dst_name is also
    provided, the new copy is queued first and then waited.
    """
    if not is_main_process():
        return None

    future = None
    if dst_name is not None:
        if root_dir is None:
            raise ValueError("root_dir is required when dst_name is provided.")
        src_path = _checkpoint_path(root_dir=root_dir, name=src_name)
        dst_path = _checkpoint_path(root_dir=root_dir, name=dst_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        suffix = uuid.uuid4().hex
        snapshot_path = f"{dst_path}.{suffix}.copy_src"
        tmp_path = f"{dst_path}.{suffix}.tmp"
        try:
            # The hard-link snapshot preserves the current checkpoint_last inode for async copy.
            os.link(src_path, snapshot_path)
        except OSError:
            shutil.copyfile(src_path, snapshot_path)
            if logger is not None:
                logger.show(
                    head="[Checkpoint] ",
                    log=f"hard-link snapshot failed; made a blocking source copy for {os.path.basename(dst_path)}.",
                    write=True,
                )

        if non_blocking:
            future = _get_checkpoint_copy_executor().submit(
                _copy_checkpoint_snapshot,
                snapshot_path,
                tmp_path,
                dst_path,
            )
            _CHECKPOINT_COPY_FUTURES.append(future)
        else:
            _copy_checkpoint_snapshot(snapshot_path=snapshot_path, tmp_path=tmp_path, dst_path=dst_path)

    if wait:
        _wait_checkpoint_copies()
    return future


def _wait_checkpoint_copies():
    global _CHECKPOINT_COPY_EXECUTOR, _CHECKPOINT_COPY_FUTURES
    if _CHECKPOINT_COPY_EXECUTOR is None:
        return

    try:
        for future in _CHECKPOINT_COPY_FUTURES:
            future.result()
    finally:
        _CHECKPOINT_COPY_FUTURES = []
        _CHECKPOINT_COPY_EXECUTOR.shutdown(wait=True)
        _CHECKPOINT_COPY_EXECUTOR = None


def load_checkpoint(
    model: nn.Module,
    path: str,
    states: dict | None = None,
    optimizer: optim.Optimizer | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    strict=True,
    report_mismatch: bool = False,
    max_mismatch_print: int = 200,
):
    load_state = torch.load(path, map_location="cpu")
    if optimizer is not None:
        optimizer.load_state_dict(load_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(load_state["scheduler"])
    if states is not None:
        states.update(load_state["states"])

    checkpoint_state_dict = load_state["model"]
    model_state_dict = model.state_dict()
    missing_keys = [k for k in model_state_dict.keys() if k not in checkpoint_state_dict]
    unexpected_keys = [k for k in checkpoint_state_dict.keys() if k not in model_state_dict]
    shape_mismatch = [
        (
            k,
            tuple(checkpoint_state_dict[k].shape),
            tuple(model_state_dict[k].shape),
        )
        for k in checkpoint_state_dict.keys()
        if k in model_state_dict and tuple(checkpoint_state_dict[k].shape) != tuple(model_state_dict[k].shape)
    ]

    if is_main_process() and report_mismatch:
        if len(missing_keys) == 0 and len(unexpected_keys) == 0 and len(shape_mismatch) == 0:
            print(f"[Checkpoint][Match] {path}")
        else:
            print(f"[Checkpoint][Mismatch] {path}")
            print(
                f"  missing_keys={len(missing_keys)}, "
                f"unexpected_keys={len(unexpected_keys)}, "
                f"shape_mismatch={len(shape_mismatch)}"
            )
            if missing_keys:
                print("  missing keys:")
                for k in missing_keys[:max_mismatch_print]:
                    print(f"    - {k}")
                if len(missing_keys) > max_mismatch_print:
                    print(f"    ... ({len(missing_keys) - max_mismatch_print} more)")
            if unexpected_keys:
                print("  unexpected keys:")
                for k in unexpected_keys[:max_mismatch_print]:
                    print(f"    - {k}")
                if len(unexpected_keys) > max_mismatch_print:
                    print(f"    ... ({len(unexpected_keys) - max_mismatch_print} more)")
            if shape_mismatch:
                print("  shape mismatch keys:")
                for k, ckpt_shape, model_shape in shape_mismatch[:max_mismatch_print]:
                    print(f"    - {k}: checkpoint={ckpt_shape}, model={model_shape}")
                if len(shape_mismatch) > max_mismatch_print:
                    print(f"    ... ({len(shape_mismatch) - max_mismatch_print} more)")

    if is_main_process():
        model.load_state_dict(checkpoint_state_dict, strict=strict)


def load_pretrained_model(model: nn.Module, pretrained_path: str, logger: Logger, show_details: bool = False):
    if not is_main_process():
        return model

    # process pretrained model path
    if not os.path.isabs(pretrained_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        weight_dir = os.path.join(project_root, 'weight')
        weight_path = os.path.join(weight_dir, pretrained_path)
        pretrained_path = weight_path if os.path.exists(weight_path) else os.path.abspath(pretrained_path)
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found at: {pretrained_path}")
    pretrained_checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    pretrained_state_dict = pretrained_checkpoint['ema']["module"]
    model_state_dict = model.state_dict()
    pretrained_keys = list(pretrained_state_dict.keys())

    if not show_details:
        logger.print("Set 'show_details=True' to see more details.\n")

    # Initialize counters for different processing branches
    adapted_count = 0  # Shape adaptation (class/score heads)
    ffn_mapped_count = 0  # FFN structure mapping
    deleted_count = 0  # Deleted parameters (anchors, etc.)
    shape_mismatch_count = 0  # Other shape mismatches

    _log = ""
    for k in pretrained_keys:
        # for denoise_emb and score_head
        if k in model_state_dict:
            if model_state_dict[k].shape != pretrained_state_dict[k].shape:
                if "score_head" in k:
                    adaptive_param, s_ref, s_org = adaptive_shape(model_state_dict[k], pretrained_state_dict[k])
                    pretrained_state_dict[k] = adaptive_param
                    adapted_count += 1
                    _log += f"Adapted {k}: {s_ref} -> {s_org} classes\n"
                elif "query_pos_head" in k:
                    pt_shape = tuple(pretrained_state_dict[k].size())
                    model_shape = tuple(model_state_dict[k].size())
                    pretrained_state_dict[k] = model_state_dict[k]
                    _log += f"Ignore query_pos_head with {k}: {pt_shape} -> {model_shape}\n"
                else:
                    shape_mismatch_count += 1
                    warning_inf = (
                        f"[Warning] Parameter {k} has shape{pretrained_state_dict[k].shape} in pretrained model, "
                        f"but get shape{model_state_dict[k].shape} in current model.\n"
                    )
                    if not show_details:
                        logger.print(warning_inf)
                    _log += warning_inf
                    # renamed FFN module in decoder layers
        elif "decoder.decoder.layers" in k and ("linear1" in k or "linear2" in k or "norm3" in k):
            new_k = k.replace("linear1", "ffn.linear1").replace("linear2", "ffn.linear2").replace("norm3", "ffn.norm")
            pretrained_state_dict[new_k] = pretrained_state_dict[k].clone()
            del pretrained_state_dict[k]
            ffn_mapped_count += 1
            _log += f"FFN mapping: {k} -> {new_k}\n"
        elif "decoder.enc_" in k or 'decoder.den' in k:
            new_k = k.replace("decoder", "decoder.query_generator")
            if new_k not in model_state_dict:
                continue
            if "score_head" in k or 'denoising_class_embed' in k:
                adaptive_param, s_ref, s_org = adaptive_shape(model_state_dict[new_k], pretrained_state_dict[k])
                pretrained_state_dict[new_k] = adaptive_param
                adapted_count += 1
                _log += f"Adapted {k}: {s_ref} -> {s_org} classes\n"
            else:
                pretrained_state_dict[new_k] = pretrained_state_dict[k].clone()
            del pretrained_state_dict[k]
        else:
            # ignore anchor concerned parameters
            # elif k in ["decoder.anchors", "decoder.valid_mask"]:
            del pretrained_state_dict[k]
            deleted_count += 1
            _log += f"Deleted {k} (will not by used)\n"

    summy = (
        f" ADAPTED CLASS/SCORE HEADS PARAMETERS (shape changed): {adapted_count}\n"
        + f" RENAMED FFN STRUCTURE PARAMETERS (renamed): {ffn_mapped_count}\n"
        + f" DELETED PARAMETERS (removed): {deleted_count}\n"
        + f" OTHER SHAPE MISMATCH PARAMETERS (unhandled): {shape_mismatch_count}\n"
    )

    _log += "\n[PARAMETER MATCHING ANALYSIS]\n"
    not_in_model = 0
    unmatched_pretrained = []
    for k in pretrained_state_dict:
        if k not in model_state_dict:
            not_in_model += 1
            unmatched_pretrained.append(k)
    _log += f" UNMATCHED PRETRAINED MODEL PARAMETERS(in pretrained model but not in current model): {not_in_model}\n"

    if show_details and unmatched_pretrained:
        for i, k in enumerate(unmatched_pretrained):
            shape = str(pretrained_state_dict[k].shape)
            _log += f"   {i + 1:3d}. {k:<50} {shape:<20}\n"

    not_in_pretrained = 0
    unmatched_current = []
    for k in model_state_dict:
        if k not in pretrained_state_dict:
            pretrained_state_dict[k] = model_state_dict[k]
            not_in_pretrained += 1
            unmatched_current.append(k)
    _log += f"\n UNMATCHED MODEL PARAMETERS(in current model but not in pretrained model): {not_in_pretrained}\n"
    if show_details and unmatched_current:
        for i, k in enumerate(unmatched_current):
            shape = str(model_state_dict[k].shape)
            _log += f"   {i + 1:3d}. {k:<50} {shape:<20}\n"

    summy += f" DROPPED PARAMETERS: {not_in_model}\n UNLOADED PARAMETERS: {not_in_pretrained}\n"
    if show_details:
        logger.show(head="load pretrained model details", log=_log + "\n" + summy, with_header=True, write=True)
    else:
        logger.show(head="load pretrained model summary", log=summy, with_header=True, write=True)
        logger.write(head="load pretrained model details:\n", log=_log)

    logger.show(head="Pretrained model is loaded from: ", log=f"{pretrained_path}", write=True)
    model.load_state_dict(state_dict=pretrained_state_dict, strict=False)
    return model


def adaptive_shape(param, param_ref):
    if param.shape[0] == 1:  # only keep person(id:0)
        adaptive_param = param_ref[:1]
    elif param.shape[0] == 2:  # person(id:0)+ background
        adaptive_param = torch.cat([param_ref[:1], param_ref[-1:]], dim=0)
    elif param.shape[0] == 8:  # BDD100K
        # We directly do not use the pretrained class embed for BDD100K
        adaptive_param = param
    else:
        raise NotImplementedError('invalid shape: {}'.format(param.shape))
        # param_ref_len = param_ref.shape[0]
        # param_len = param.shape[0]
        # adaptive_param = torch.cat([param_ref[:param_len-1], param_ref[-1]], dim=0)

    return adaptive_param, param_ref.shape[0], param.shape[0]


def logits_to_scores(logits: torch.Tensor):
    return logits.sigmoid()


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def Debug_print(t: torch.Tensor):
    print(f"[DEBUG] ConvNormLayer input x: shape={t.shape}, dtype={t.dtype}")
    print(
        f"[DEBUG] ConvNormLayer input x: min={t.min().item():.6f}, max={t.max().item():.6f}, mean={t.mean().item():.6f}"
    )
    print(
        f"[DEBUG] ConvNormLayer input x: has_nan={torch.isnan(t).any().item()}, has_inf={torch.isinf(t).any().item()}"
    )
