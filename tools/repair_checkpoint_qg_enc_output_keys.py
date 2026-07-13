#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename HybridQueryGenerator enc_output keys in checkpoint."
    )
    parser.add_argument("--src", type=Path, required=True, help="Source checkpoint path.")
    parser.add_argument("--dst", type=Path, required=True, help="Destination checkpoint path.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {args.src}")

    mapping = {
        "decoder.query_generator.enc_output.0.weight": "decoder.query_generator.enc_output.proj.weight",
        "decoder.query_generator.enc_output.0.bias": "decoder.query_generator.enc_output.proj.bias",
        "decoder.query_generator.enc_output.1.weight": "decoder.query_generator.enc_output.norm.weight",
        "decoder.query_generator.enc_output.1.bias": "decoder.query_generator.enc_output.norm.bias",
    }

    ckpt = torch.load(str(args.src), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must be dict, got {type(ckpt)}")
    if "model" not in ckpt or not isinstance(ckpt["model"], dict):
        raise KeyError("Checkpoint does not contain a dict field 'model'.")

    model_state = ckpt["model"]
    renamed = []
    skipped = []
    for old_k, new_k in mapping.items():
        if old_k not in model_state:
            skipped.append((old_k, new_k, "old_key_missing"))
            continue
        if new_k in model_state:
            skipped.append((old_k, new_k, "new_key_already_exists"))
            continue
        model_state[new_k] = model_state.pop(old_k)
        renamed.append((old_k, new_k))

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(args.dst))

    print(f"[Done] saved: {args.dst}")
    print(f"[Summary] renamed={len(renamed)}, skipped={len(skipped)}")
    if renamed:
        print("[Renamed]")
        for old_k, new_k in renamed:
            print(f"  {old_k} -> {new_k}")
    if skipped:
        print("[Skipped]")
        for old_k, new_k, reason in skipped:
            print(f"  {old_k} -> {new_k} ({reason})")


if __name__ == "__main__":
    main()
