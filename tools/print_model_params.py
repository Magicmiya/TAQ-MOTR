#!/usr/bin/env python3
"""
Load a TAQ_MOTR checkpoint and print parameter counts per top-level submodule.
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print parameter counts per top-level module from a TAQ_MOTR checkpoint."
    )
    parser.add_argument(
        "-F",
        "--file_path",
        type=Path,
        required=True,
        help="Path to the checkpoint file.",
    )
    return parser.parse_args()


# Suffixes that identify batch-norm running statistics (not trainable parameters)
BN_RUNNING_STATS_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked")


def count_parameters(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(v.numel() for v in state_dict.values())


def is_bn_stat(key: str) -> bool:
    return key.endswith(BN_RUNNING_STATS_SUFFIXES)


def group_by_top_module(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, dict[str, torch.Tensor]]:
    """Group state_dict entries by their top-level module prefix, excluding BN running stats."""
    groups: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
    other: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if is_bn_stat(key):
            continue
        if "." in key:
            top_module = key.split(".")[0]
        else:
            other[key] = tensor
            continue

        if top_module not in groups:
            groups[top_module] = {}
        groups[top_module][key] = tensor

    if other:
        groups["__other__"] = other

    return groups


def print_param_report(state_dict: dict[str, torch.Tensor]):
    """Print a report of trainable parameter counts, excluding BN running statistics."""

    # Total excluding running stats = trainable parameters
    trainable_total = sum(v.numel() for k, v in state_dict.items() if not is_bn_stat(k))
    running_stats_total = sum(v.numel() for k, v in state_dict.items() if is_bn_stat(k))
    grand_total = trainable_total + running_stats_total

    groups = group_by_top_module(state_dict)

    print()
    print("=" * 72)
    print(f"{'Submodule':<30} {'Trainable Params':>20} {'% of Total':>10}")
    print("=" * 72)

    sum_grouped = 0
    for name, group in groups.items():
        n_params = count_parameters(group)
        sum_grouped += n_params
        pct = 100.0 * n_params / trainable_total if trainable_total > 0 else 0.0
        if n_params >= 1_000_000:
            print(f"{name:<30} {n_params:>15,} ({n_params/1e6:.2f}M)  {pct:>8.2f}%")
        elif n_params >= 1_000:
            print(f"{name:<30} {n_params:>15,} ({n_params/1e3:.1f}K)   {pct:>8.2f}%")
        else:
            print(f"{name:<30} {n_params:>15,}                    {pct:>8.2f}%")

    print("=" * 72)

    if sum_grouped == trainable_total:
        print(f"{'Total (trainable)':<30} {trainable_total:>15,} ({trainable_total/1e6:.2f}M)  {100.0:>8.2f}%")
    else:
        print(f"{'Total (grouped)':<30} {sum_grouped:>15,} ({sum_grouped/1e6:.2f}M)")
        print(f"{'Total (trainable)':<30} {trainable_total:>15,} ({trainable_total/1e6:.2f}M)  {100.0:>8.2f}%")

    if running_stats_total > 0:
        print(f"\n  + {running_stats_total:,} ({running_stats_total/1e6:.2f}M) BN running statistics (non-trainable)")
        print(f"  = {grand_total:,} ({grand_total/1e6:.2f}M) total tensors in state_dict")

    print(f"\nTotal trainable parameters: {trainable_total:,} ({trainable_total / 1e6:.2f}M)")


def main():
    args = parse_args()
    file_path = args.file_path.expanduser().resolve()

    print(f"Loading checkpoint: {file_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}", file=sys.stderr)
        sys.exit(1)

    if isinstance(ckpt, dict):
        print(f"Checkpoint is a dict with keys: {list(ckpt.keys())}")

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
            print(f"Found state_dict under 'model' key with {len(state_dict)} entries.")
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
            print(f"Found state_dict under 'state_dict' key with {len(state_dict)} entries.")
        elif "ema" in ckpt and isinstance(ckpt["ema"], dict):
            state_dict = ckpt["ema"]
            print(f"Found state_dict under 'ema' key with {len(state_dict)} entries.")
        else:
            candidate = None
            for k, v in ckpt.items():
                if isinstance(v, dict) and len(v) > 100:
                    sample_keys = list(v.keys())[:5]
                    if any("." in key for key in sample_keys):
                        candidate = v
                        print(f"Guessing state_dict under '{k}' key ({len(v)} entries, sample keys: {sample_keys})")
                        break
            if candidate is not None:
                state_dict = candidate
            else:
                print("ERROR: Could not find state_dict in checkpoint.", file=sys.stderr)
                sys.exit(1)
    elif hasattr(ckpt, "state_dict"):
        print("Checkpoint is a full model object.")
        state_dict = ckpt.state_dict()
    else:
        print(f"ERROR: Unrecognized checkpoint format: {type(ckpt)}", file=sys.stderr)
        sys.exit(1)

    print_param_report(state_dict)


if __name__ == "__main__":
    main()
