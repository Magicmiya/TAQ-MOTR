#!/usr/bin/env python3
import argparse
import uuid
from collections.abc import Mapping
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a checkpoint that only contains model weights."
    )
    parser.add_argument("checkpoint", nargs="?", type=Path, help="Source checkpoint path.")
    parser.add_argument("--src", type=Path, default=None, help="Source checkpoint path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing lite checkpoint.")
    return parser.parse_args()


def resolve_src(args) -> Path:
    if args.checkpoint is None and args.src is None:
        raise ValueError("Please provide a checkpoint path, either positionally or with --src.")
    if args.checkpoint is not None and args.src is not None and args.checkpoint != args.src:
        raise ValueError("Got two different checkpoint paths from positional argument and --src.")
    return args.src if args.src is not None else args.checkpoint


def make_lite_path(src: Path) -> Path:
    if src.suffix:
        return src.with_name(f"{src.stem}_lite{src.suffix}")
    return src.with_name(f"{src.name}_lite.pth")


def torch_load_cpu(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def extract_model_weights(ckpt):
    if not isinstance(ckpt, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(ckpt)}.")

    for key in ("model", "state_dict", "model_state_dict"):
        if key not in ckpt:
            continue
        model_state = ckpt[key]
        if not isinstance(model_state, Mapping):
            raise TypeError(f"Checkpoint field '{key}' must be a mapping, got {type(model_state)}.")
        return model_state, key

    if all(torch.is_tensor(value) for value in ckpt.values()):
        return ckpt, "<root>"

    raise KeyError("Checkpoint does not contain 'model', 'state_dict', or pure root-level weights.")


def summarize_weights(model_state: Mapping) -> dict:
    tensors = [value for value in model_state.values() if torch.is_tensor(value)]
    return {
        "keys": len(model_state),
        "tensors": len(tensors),
        "numel": sum(tensor.numel() for tensor in tensors),
    }


def save_atomic(obj, dst: Path):
    tmp_path = dst.with_name(f"{dst.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(obj, str(tmp_path))
        tmp_path.replace(dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main():
    args = parse_args()
    src = resolve_src(args)
    dst = make_lite_path(src)

    if not src.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src}")
    if src.resolve() == dst.resolve():
        raise ValueError(f"Output path must differ from input path: {dst}")
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"Lite checkpoint already exists: {dst}. Use --overwrite to replace it.")

    ckpt = torch_load_cpu(src)
    model_state, source_key = extract_model_weights(ckpt)
    summary = summarize_weights(model_state)

    # Save only the extracted weight mapping; optimizer, scheduler, and training states are intentionally dropped.
    save_atomic(model_state, dst)

    print(f"[Input] {src}")
    print(f"[Output] {dst}")
    print(f"[Source] {source_key}")
    print(
        "[Summary] "
        f"keys={summary['keys']}, tensors={summary['tensors']}, numel={summary['numel']}"
    )


if __name__ == "__main__":
    main()
