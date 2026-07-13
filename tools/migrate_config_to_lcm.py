#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate legacy TAQ_MOTR yaml config to Life_cycle_management layout.")
    parser.add_argument("--src", type=Path, required=True, help="Source yaml path")
    parser.add_argument("--dst", type=Path, default=None, help="Destination yaml path (default: overwrite src)")
    parser.add_argument(
        "--prune-criterion-fields",
        action="store_true",
        help="Remove lifecycle fields from Criterion after migration",
    )
    return parser.parse_args()


def get_nested(cfg: dict[str, Any], k1: str, k2: str, default: Any) -> Any:
    d = cfg.get(k1, {})
    if isinstance(d, dict):
        return d.get(k2, default)
    return default


def migrate(cfg: dict[str, Any], prune_criterion_fields: bool = False) -> tuple[dict[str, Any], list[str]]:
    logs: list[str] = []

    criterion = cfg.get("Criterion", {}) if isinstance(cfg.get("Criterion", {}), dict) else {}
    q_updater = cfg.get("Query_updater", {}) if isinstance(cfg.get("Query_updater", {}), dict) else {}

    lcm = cfg.get("Life_cycle_management", {}) if isinstance(cfg.get("Life_cycle_management", {}), dict) else {}

    def set_default(key: str, value: Any):
        if key not in lcm:
            lcm[key] = value
            logs.append(f"set Life_cycle_management.{key}={value}")

    set_default("hidden_dim", get_nested(cfg, "Query_updater", "hidden_dim", get_nested(cfg, "Criterion", "hidden_dim", 256)))
    set_default("num_classes", get_nested(cfg, "Criterion", "num_classes", cfg.get("num_classes", 1)))
    set_default("high_conf_threshold", get_nested(cfg, "Criterion", "high_conf_threshold", 0.5))
    set_default("new_born_threshold", get_nested(cfg, "Criterion", "new_born_threshold", 0.9))
    set_default("track_thresh", get_nested(cfg, "Criterion", "track_thresh", 0.5))
    set_default("miss_tolerance", get_nested(cfg, "Criterion", "miss_tolerance", 30))
    set_default("update_threshold", get_nested(cfg, "Query_updater", "update_threshold", 0.5))
    set_default("long_memory_lambda", get_nested(cfg, "Query_updater", "long_memory_lambda", 0.01))
    set_default("tp_drop_ratio", get_nested(cfg, "Query_updater", "tp_drop_ratio", 0.0))
    set_default("fp_insert_ratio", get_nested(cfg, "Query_updater", "fp_insert_ratio", 0.0))
    set_default("no_tracking_augment", get_nested(cfg, "Query_updater", "no_tracking_augment", True))
    set_default("recover_iou_threshold", 0.5)

    cfg["Life_cycle_management"] = lcm

    if prune_criterion_fields and isinstance(criterion, dict):
        remove_keys = ["high_conf_threshold", "new_born_threshold", "track_thresh", "miss_tolerance"]
        for k in remove_keys:
            if k in criterion:
                criterion.pop(k)
                logs.append(f"remove Criterion.{k}")
        cfg["Criterion"] = criterion

    return cfg, logs


def main() -> None:
    args = parse_args()
    src = args.src
    dst = args.dst if args.dst is not None else args.src

    if not src.exists():
        raise FileNotFoundError(f"Config file not found: {src}")

    with src.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if not isinstance(cfg, dict):
        raise TypeError(f"yaml root must be dict, got {type(cfg)}")

    migrated, logs = migrate(cfg, prune_criterion_fields=args.prune_criterion_fields)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        yaml.safe_dump(migrated, f, sort_keys=False, allow_unicode=True)

    print(f"[Done] saved: {dst}")
    if not logs:
        print("[Summary] no changes")
    else:
        print(f"[Summary] {len(logs)} changes")
        for line in logs:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
