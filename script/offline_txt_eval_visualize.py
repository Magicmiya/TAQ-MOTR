from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MPLCONFIGDIR = REPO_ROOT / "tmp" / "matplotlib-offline-txt"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import numpy as np
import yaml
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None

from data import build_dataset
from script import evaluator as eval_utils
from utils.visualizer import Visualizer

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing MOT txt results and generate bbox-rendered images without model inference.",
    )
    parser.add_argument(
        "-C",
        "--config_path",
        default="configs/dancetrack_train.yaml",
        help="Dataset main config used for dataset and visualizer defaults.",
    )
    parser.add_argument(
        "-DR",
        "--dataset_root",
        required=True,
        help="Dataset root containing DanceTrack, MOT17, MOT20, or SportMOT.",
    )
    parser.add_argument(
        "-T",
        "--tracker_dir",
        required=True,
        help="Tracker result directory, for example: outputs/MeMOTR/val/inference_val/memotr_ddetr_dancetrack",
    )
    parser.add_argument(
        "-M",
        "--mode",
        default="val",
        choices=["val", "test"],
        help="Dataset split to use.",
    )
    parser.add_argument(
        "-V",
        "--videos",
        default="",
        help="Comma-separated video subset, for example: dancetrack0041,dancetrack0043",
    )
    return parser.parse_args()


def _load_yaml_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.load(f.read(), yaml.FullLoader)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config at {config_path} must load as a dict, got: {type(cfg)}")
    return cfg


def _parse_video_filter(raw_videos: str) -> list[str] | None:
    normalized = [video.strip() for video in str(raw_videos).split(",") if video.strip()]
    return normalized or None


def _resolve_tracker_layout(tracker_dir: str, mode: str) -> dict[str, str]:
    tracker_path = Path(tracker_dir).expanduser().resolve()
    if not tracker_path.is_dir():
        raise FileNotFoundError(f"Tracker directory does not exist: {tracker_path}")

    trackers_dir = tracker_path.parent
    mode_dir = trackers_dir.parent
    outputs_dir = mode_dir.parent
    if mode_dir.name != mode:
        raise ValueError(
            f"Tracker directory mode mismatch: expected parent folder '{mode}', got '{mode_dir.name}' from {tracker_path}"
        )

    return {
        "tracker_dir": str(tracker_path),
        "trackers_to_eval": trackers_dir.name,
        "tracker_sub_folder": tracker_path.name,
        "outputs_dir": str(outputs_dir),
    }


def _build_runtime_config(base_cfg: dict, args: argparse.Namespace, layout: dict[str, str]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["MODE"] = args.mode
    cfg["OUTPUTS_DIR"] = layout["outputs_dir"]
    cfg["visualize"] = True

    dataset_cfg = dict(cfg.get("Dataset", {}))
    # Main configs intentionally leave environment-specific dataset paths empty.
    dataset_cfg["dataset_root"] = args.dataset_root
    dataset_cfg["eval_videos"] = _parse_video_filter(args.videos)
    cfg["Dataset"] = dataset_cfg
    return cfg


def _build_bbox_visualizer_cfg(cfg: dict) -> dict:
    visualizer_cfg = copy.deepcopy(cfg.get("Visualizer", {}))
    tasks_cfg = dict(visualizer_cfg.get("tasks", {}))

    bbox_cfg = dict(tasks_cfg.get("bbox_render", {}))
    bbox_cfg["enabled"] = True
    bbox_cfg["save_image"] = True
    bbox_cfg["show_image"] = False
    tasks_cfg["bbox_render"] = bbox_cfg

    for task_name, task_cfg in list(tasks_cfg.items()):
        if task_name == "bbox_render":
            continue
        disabled_cfg = dict(task_cfg)
        disabled_cfg["enabled"] = False
        tasks_cfg[task_name] = disabled_cfg

    # Keep offline rendering side effects narrow to bbox images only.
    visualizer_cfg["enabled"] = True
    visualizer_cfg["workers"] = int(visualizer_cfg.get("workers", 4))
    visualizer_cfg["writer_workers"] = int(visualizer_cfg.get("writer_workers", 2))
    visualizer_cfg["hook"] = {"enabled": False}
    visualizer_cfg["time"] = {"enabled": False}
    visualizer_cfg["tasks"] = tasks_cfg
    return visualizer_cfg

def _build_visualize_root_name(tracker_sub_folder: str) -> str:
    """Build a stable visualize directory name for one evaluated result."""
    # Match online evaluation naming so offline multi-run rendering does not overwrite previous outputs.
    return f"visualize_{tracker_sub_folder}"


def _load_track_results_by_frame(result_path: Path) -> dict[int, np.ndarray]:
    if not result_path.is_file():
        raise FileNotFoundError(f"Tracker result file does not exist: {result_path}")
    if result_path.stat().st_size == 0:
        return {}

    result = np.loadtxt(result_path, delimiter=",", ndmin=2, dtype=np.float32)
    if result.size == 0:
        return {}

    frame_ids = result[:, 0].astype(np.int32)
    grouped: dict[int, np.ndarray] = {}
    for frame_id in np.unique(frame_ids):
        grouped[int(frame_id)] = np.ascontiguousarray(result[frame_ids == frame_id])
    return grouped


def _iter_video_frames(video_dir: Path) -> list[tuple[int, str]]:
    img_dir = video_dir / "img1"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {img_dir}")

    frames: list[tuple[int, str]] = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            frame_id = int(img_path.stem)
        except ValueError:
            continue
        frames.append((frame_id, str(img_path)))

    if not frames:
        raise RuntimeError(f"No image frames found under: {img_dir}")
    return frames


def _collect_video_frames(dataset) -> tuple[list[str], dict[str, list[tuple[int, str]]], int]:
    selected_videos = dataset.get_selected_videos(mode=dataset.mode)
    frames_by_video: dict[str, list[tuple[int, str]]] = {}
    total_frames = 0
    for video_name in selected_videos:
        video_dir = Path(dataset.split_dir) / video_name
        video_frames = _iter_video_frames(video_dir)
        frames_by_video[video_name] = video_frames
        total_frames += len(video_frames)
    return selected_videos, frames_by_video, total_frames


def _render_bbox_images(dataset, tracker_dir: Path, visualizer: Visualizer) -> tuple[int, int]:
    empty_track = np.empty((0, 9), dtype=np.float32)
    selected_videos, frames_by_video, total_frames = _collect_video_frames(dataset)
    rendered_videos = 0
    rendered_frames = 0
    progress = None

    if tqdm is not None:
        progress = tqdm(
            total=total_frames,
            desc=f"Offline bbox render {dataset.mode}",
            unit="it",
            dynamic_ncols=True,
        )
    else:
        print(f"Offline bbox render {dataset.mode}: 0/{total_frames}")

    # Render every source frame so empty-result frames are still visually inspectable.
    for video_name in selected_videos:
        track_path = tracker_dir / f"{video_name}.txt"
        frame_results = _load_track_results_by_frame(track_path)

        for frame_id, img_path in frames_by_video[video_name]:
            visualizer.update(
                batch={"infos": {"frame_idx": frame_id, "video_name": video_name, "img_path": img_path}},
                track_result=frame_results.get(frame_id, empty_track),
            )
            rendered_frames += 1
            if progress is not None:
                progress.set_postfix_str(f"{video_name}={frame_id:04d}", refresh=False)
                progress.update(1)
            elif rendered_frames == total_frames or rendered_frames % 200 == 0:
                print(f"Offline bbox render {dataset.mode}: {rendered_frames}/{total_frames} [{video_name}={frame_id:04d}]")

        rendered_videos += 1

    if progress is not None:
        progress.close()
    return rendered_videos, rendered_frames


def main() -> None:
    args = parse_args()
    base_cfg = _load_yaml_config(args.config_path)
    layout = _resolve_tracker_layout(args.tracker_dir, args.mode)
    cfg = _build_runtime_config(base_cfg=base_cfg, args=args, layout=layout)

    dataset = build_dataset(config=cfg["Dataset"], mode=args.mode)
    selected_videos = dataset.get_selected_videos(mode=args.mode)
    print(f"Tracker dir: {layout['tracker_dir']}")
    print(f"Selected videos ({len(selected_videos)}): {', '.join(selected_videos)}")

    summary, report = eval_utils.evaluate(
        cfg,
        dataset,
        trackers_to_eval=layout["trackers_to_eval"],
        tracker_sub_folder=layout["tracker_sub_folder"],
    )

    visualizer_cfg = _build_bbox_visualizer_cfg(cfg)
    visualizer_cfg["root_dir_name"] = _build_visualize_root_name(layout["tracker_sub_folder"])
    visualizer = Visualizer(
        cfg=visualizer_cfg,
        mode=args.mode,
        save_path=cfg["OUTPUTS_DIR"],
    )
    try:
        rendered_videos, rendered_frames = _render_bbox_images(
            dataset=dataset,
            tracker_dir=Path(layout["tracker_dir"]),
            visualizer=visualizer,
        )
    finally:
        visualizer.close()

    eval_text_lines = [
        f"Tracker dir: {layout['tracker_dir']}",
        f"Tracker group: {layout['trackers_to_eval']}",
        f"Rendered videos: {rendered_videos}",
        f"Rendered frames: {rendered_frames}",
    ]
    summary_text = eval_utils._format_eval_summary(summary)
    if summary_text:
        eval_text_lines.extend(["", summary_text])
    eval_utils.write_default_eval_result(
        config=cfg,
        eval_split=args.mode,
        trackers_to_eval=layout["trackers_to_eval"],
        sub_out_path=layout["tracker_sub_folder"],
        eval_msg="\n".join(eval_text_lines),
        trackeval_stdout=report.get("stdout", ""),
        formatted_tsv=report.get("formatted_tsv", ""),
    )
    eval_utils.copy_eval_result_to_visualize(
        config=cfg,
        eval_split=args.mode,
        trackers_to_eval=layout["trackers_to_eval"],
        sub_out_path=layout["tracker_sub_folder"],
    )

    print(f"Rendered bbox images to: {Path(cfg['OUTPUTS_DIR']) / _build_visualize_root_name(layout['tracker_sub_folder'])}")
    if summary_text:
        print("\n" + summary_text)


if __name__ == "__main__":
    main()
