from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MPLCONFIGDIR = REPO_ROOT / "tmp" / "matplotlib-offline-vis"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import yaml

from data import build_dataset
from utils.visualizer.tasks import MultiTrackerCompareTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline multi-tracker qualitative compare render with GT-aligned IDSW/Lost status analysis.",
    )
    parser.add_argument("-C", "--config_path", default="configs/dancetrack_train.yaml")
    parser.add_argument(
        "-DR",
        "--dataset_root",
        required=True,
        help="Dataset root containing DanceTrack, MOT17, MOT20, or SportMOT.",
    )
    parser.add_argument("-M", "--mode", default="val", choices=["val", "test"])
    parser.add_argument("-V", "--videos", default="", help="Comma-separated video subset.")
    parser.add_argument(
        "--tracker_dirs",
        nargs=3,
        required=True,
        metavar=("TRACKER_A", "TRACKER_B", "TRACKER_C"),
        help="Three tracker result directories. The last tracker is used as primary method in compare mode.",
    )
    parser.add_argument(
        "--tracker_labels",
        nargs=3,
        default=("Generative", "Learnable", "HQG"),
        metavar=("LABEL_A", "LABEL_B", "LABEL_C"),
    )
    parser.add_argument("--output_dir", default="outputs/results/DanceTrack/multi_overlay")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_compare", action="store_true", help="Disable keyframe compare mode and render all frames.")
    parser.add_argument("--context_frames", type=int, default=5)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--focus_gt_id", default=None, help="Render only selected GT ids, for example: 3 or 3,4. This overrides compare keyframe filtering.")
    parser.add_argument("--search_stable_clip", action="store_true", help="Search stable HQG clips with switched Generate/Learnable ids and render them.")
    parser.add_argument("--search_window", type=int, default=20)
    parser.add_argument("--search_limit", type=int, default=-1, help="Maximum number of searched clips to keep. Use -1 to keep all.")
    parser.add_argument("--show_related_idsw_boxes", action="store_true", help="Draw idsw-related prediction boxes with tracker ids.")
    parser.add_argument("--related_box_mode", default="idsw", choices=["idsw"])
    parser.add_argument(
        "--target_frames",
        default="",
        help="Explicit frames to render, for example: dancetrack0019:2037,2046;dancetrack0030:1206",
    )
    parser.add_argument("--max_frames", type=int, default=0, help="Optional debug cap for rendered frames per video.")
    parser.add_argument("--max_table_rows", type=int, default=8)
    parser.add_argument(
        "--compare_statuses",
        nargs="+",
        default=("idsw", "lost"),
        help="Statuses in non-primary methods which trigger keyframe selection.",
    )
    return parser.parse_args()


def _load_yaml_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.load(f.read(), yaml.FullLoader)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config at {config_path} must load as a dict, got: {type(cfg)}")
    return cfg


def _parse_video_filter(raw_videos: str) -> list[str] | None:
    videos = [video.strip() for video in str(raw_videos).split(",") if video.strip()]
    return videos or None


def _parse_focus_gt_ids(raw_focus_gt_id) -> list[int] | None:
    if raw_focus_gt_id in (None, "", "None"):
        return None
    if isinstance(raw_focus_gt_id, int):
        gt_ids = [raw_focus_gt_id]
    else:
        gt_ids = [int(token.strip()) for token in str(raw_focus_gt_id).split(",") if token.strip()]
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for gt_id in gt_ids:
        if gt_id in seen:
            continue
        seen.add(gt_id)
        ordered_unique.append(gt_id)
    return ordered_unique or None


def _parse_target_frames(raw_target_frames: str) -> dict[str, list[int]]:
    target_frames: dict[str, list[int]] = {}
    for group in str(raw_target_frames).split(";"):
        group = group.strip()
        if not group:
            continue
        if ":" not in group:
            raise ValueError(f"Invalid target frame group: {group}")
        video_name, raw_frames = group.split(":", 1)
        video_name = video_name.strip()
        if not video_name:
            raise ValueError(f"Invalid target frame video name: {group}")
        frames = [int(token.strip()) for token in raw_frames.split(",") if token.strip()]
        if not frames:
            raise ValueError(f"No frames provided for target video: {video_name}")
        ordered_unique: list[int] = []
        seen: set[int] = set()
        for frame_id in frames:
            if frame_id <= 0:
                raise ValueError(f"Target frame ids must be positive, got {frame_id} in {video_name}")
            if frame_id in seen:
                continue
            seen.add(frame_id)
            ordered_unique.append(frame_id)
        target_frames[video_name] = ordered_unique
    return target_frames


def _build_runtime_config(base_cfg: dict, args: argparse.Namespace) -> dict:
    cfg = dict(base_cfg)
    cfg["MODE"] = args.mode
    dataset_cfg = dict(cfg.get("Dataset", {}))
    # Main configs intentionally leave environment-specific dataset paths empty.
    dataset_cfg["dataset_root"] = args.dataset_root
    dataset_cfg["eval_videos"] = _parse_video_filter(args.videos)
    cfg["Dataset"] = dataset_cfg
    return cfg


def _build_task_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    del cfg
    return {
        "compare": not bool(args.no_compare),
        "iou_threshold": float(args.iou_threshold),
        "context_frames": int(args.context_frames),
        "primary_tracker_index": -1,
        "primary_require_match": True,
        "compare_statuses": list(args.compare_statuses),
        "focus_gt_ids": _parse_focus_gt_ids(args.focus_gt_id),
        "search_stable_clip": bool(args.search_stable_clip),
        "search_window": int(args.search_window),
        "search_limit": int(args.search_limit),
        "show_related_idsw_boxes": bool(args.show_related_idsw_boxes or args.search_stable_clip),
        "related_box_mode": str(args.related_box_mode),
        "target_frames_by_video": _parse_target_frames(args.target_frames),
        "tracker_dirs": list(args.tracker_dirs),
        "tracker_labels": list(args.tracker_labels),
        "output_dir": str(args.output_dir),
        "overwrite": bool(args.overwrite),
        "border_alpha": 0.92,
        "line_thickness": 3,
        "label_font_scale": 0.44,
        "anchor_text_step": 16,
        "max_table_rows": int(args.max_table_rows),
        "max_frames": int(args.max_frames),
        "render_non_key_targets": False,
    }


def main() -> None:
    args = parse_args()
    try:
        base_cfg = _load_yaml_config(args.config_path)
        runtime_cfg = _build_runtime_config(base_cfg=base_cfg, args=args)
        dataset = build_dataset(config=runtime_cfg["Dataset"], mode=args.mode)
    except Exception as exc:
        print(f"[Error警告] offline compare setup failed: {exc}")
        return

    task_cfg = _build_task_cfg(cfg=runtime_cfg, args=args)
    task = MultiTrackerCompareTask(cfg=task_cfg)
    task.run(dataset=dataset)


if __name__ == "__main__":
    main()
