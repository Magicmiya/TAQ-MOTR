from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys
import zipfile
from typing import Iterable, Iterator, TypeVar

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MPLCONFIGDIR = REPO_ROOT / "tmp" / "matplotlib-query-focus"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import cv2
import numpy as np
import torch
import yaml

from configs.utils import (
    _merge_eval_with_train_model_config,
    _missing_model_config_keys,
    _try_load_checkpoint_train_config,
    get_git_version,
)
from script.inference import _build_eval_sub_out_path, _resolve_inference_root, inference_offline

SUMMARY_FILENAME = "video_summary.json"
BUNDLE_FILENAME = "video_bundle.zip"
BUNDLE_SUMMARY_FILENAME = "summary.json"
_T = TypeVar("_T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline query-focus visualization entrypoint that runs multiple checkpoints and composes panels.",
    )
    parser.add_argument("-C", "--config_path", default="configs/dancetrack_train.yaml")
    parser.add_argument(
        "-DR",
        "--dataset_root",
        required=True,
        help="Dataset root containing DanceTrack, MOT17, MOT20, or SportMOT.",
    )
    parser.add_argument("-M", "--mode", default="val", choices=["val"])
    parser.add_argument("-V", "--videos", default="", help="Comma-separated video subset. Empty means the full val split.")
    parser.add_argument(
        "--method_ckpt_dirs",
        nargs=3,
        required=True,
        metavar=("GEN_CKPT", "LEARN_CKPT", "HQG_CKPT"),
        help="Three checkpoint roots or checkpoint_last.pth files.",
    )
    parser.add_argument(
        "--method_labels",
        nargs=3,
        default=("Generative", "Learnable", "HQG"),
        metavar=("GEN_LABEL", "LEARN_LABEL", "HQG_LABEL"),
    )
    parser.add_argument(
        "--method_num_queries",
        nargs=3,
        type=int,
        default=(300, 300, 200),
        metavar=("GEN_Q", "LEARN_Q", "HQG_Q"),
    )
    parser.add_argument(
        "--method_hqg_memories",
        nargs=3,
        type=int,
        default=(64, 64, 32),
        metavar=("GEN_MEM", "LEARN_MEM", "HQG_MEM"),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/results/DanceTrack",
        help="Parent directory under which one multi_focus_xxxx run directory will be created.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the composed panel directory if it exists.")
    parser.add_argument("--skip_compose", action="store_true", help="Only run inference and skip panel composition.")
    parser.add_argument("--compose_only", action="store_true", help="Skip inference and compose from existing visualize outputs.")
    parser.add_argument(
        "--task_name",
        default="decoder_l0_query_focus",
        help="Task subdirectory name to compose under each visualize root.",
    )
    parser.add_argument(
        "--compose_header_mode",
        default="compose",
        choices=["task", "compose", "none"],
        help="Whether frame headers are rendered by task outputs, by compose, or disabled completely.",
    )
    parser.add_argument(
        "--crop_scope",
        default="none",
        choices=["none", "frame", "video"],
        help="Crop scope driven by GT boxes. Use none to disable cropping.",
    )
    parser.add_argument(
        "--crop_margin",
        type=int,
        default=50,
        help="Pixel margin added outside the GT union box when crop_scope is not none.",
    )
    parser.set_defaults(include_original=True)
    parser.add_argument(
        "--include_original",
        dest="include_original",
        action="store_true",
        help="Keep the original image as the first panel.",
    )
    parser.add_argument(
        "--exclude_original",
        dest="include_original",
        action="store_false",
        help="Compose only the compared methods without the original image.",
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


def _slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    return text.strip("_") or "method"


def _build_video_tag(videos: list[str] | None, mode: str) -> str:
    if not videos:
        return f"all_{mode}"
    if len(videos) == 1:
        return _slugify(videos[0])
    return f"subset_{len(videos)}"


def _build_run_root(args: argparse.Namespace) -> Path:
    videos = _parse_video_filter(args.videos)
    video_tag = _build_video_tag(videos=videos, mode=str(args.mode))
    return Path(args.output_dir).expanduser().resolve() / f"multi_focus_{video_tag}"


def _resolve_checkpoint_path(path: str) -> Path:
    raw_path = Path(path).expanduser().resolve()
    if raw_path.is_file():
        if raw_path.suffix != ".pth":
            raise ValueError(f"Expected a .pth checkpoint file, got: {raw_path}")
        return raw_path
    if raw_path.is_dir():
        ckpt_path = raw_path / "checkpoint_last.pth"
        if ckpt_path.is_file():
            return ckpt_path
        raise FileNotFoundError(f"checkpoint_last.pth not found under: {raw_path}")
    raise FileNotFoundError(f"Checkpoint path does not exist: {raw_path}")


def _build_exp_name(label: str, videos: list[str] | None) -> str:
    base = f"DanceTrack_QFocus_{_slugify(label)}"
    if not videos:
        return base
    if len(videos) == 1:
        match = re.search(r"(\d+)$", str(videos[0]))
        if match is not None:
            return f"{base}_smoke{match.group(1)}"
        return f"{base}_{_slugify(videos[0])}"
    return f"{base}_{_build_video_tag(videos=videos, mode='val')}"


def _configure_focus_task(visualizer_cfg: dict, compose_header_mode: str) -> dict:
    tasks_cfg = dict(visualizer_cfg.get("tasks", {}))
    disabled_tasks = {}
    for task_name, task_cfg in tasks_cfg.items():
        task_copy = dict(task_cfg)
        task_copy["enabled"] = False
        disabled_tasks[task_name] = task_copy
    focus_defaults = {
        "save_image": True,
        "show_image": False,
        "window_delay": 1,
        "alpha": 0.42,
        "min_score": 0.0,
        "topk_queries": 200,
        "aggregate_mode": "sum",
        "level_weight_mode": "uniform",
        "blur_kernel": 21,
        "blur_sigma": 0.0,
        "norm_percentile": 99.0,
        "draw_prev_track_bbox": True,
        "draw_missing_gt_bbox": True,
        "prev_track_bbox_alpha": 0.6,
        "prev_track_bbox_thickness": 3,
        "missing_gt_bbox_thickness": 3,
        "missing_gt_iou_threshold": 0.5,
        "gt_visibility_threshold": 0.0,
    }
    focus_cfg = dict(focus_defaults)
    focus_cfg.update(disabled_tasks.get("decoder_l0_query_focus", tasks_cfg.get("decoder_l0_query_focus", {})))
    focus_cfg["enabled"] = True
    if compose_header_mode == "task":
        focus_cfg["draw_frame_text"] = True
    else:
        focus_cfg["draw_frame_text"] = False
    disabled_tasks["decoder_l0_query_focus"] = focus_cfg
    visualizer_cfg["tasks"] = disabled_tasks
    visualizer_cfg["workers"] = 8
    visualizer_cfg["writer_workers"] = 4
    hook_cfg = dict(visualizer_cfg.get("hook", {}))
    hook_cfg["enabled"] = True
    hook_cfg["queue_size"] = 8192
    hook_cfg["block_on_full"] = True
    hook_cfg["clone_tensor"] = True
    visualizer_cfg["hook"] = hook_cfg
    return visualizer_cfg


def _prepare_runtime_config(
    base_cfg: dict,
    args: argparse.Namespace,
    run_root: Path,
    label: str,
    checkpoint_path: Path,
    num_queries: int,
    hqg_memory: int,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["MODE"] = str(args.mode)
    cfg["EVAL_FILE_PATH"] = str(checkpoint_path)
    cfg["EXP_NAME"] = _build_exp_name(label=label, videos=_parse_video_filter(args.videos))
    cfg["OUTPUTS_DIR"] = str(checkpoint_path.parent)
    cfg["GIT_VERSION"] = get_git_version()
    cfg["visualize"] = True

    dataset_cfg = dict(cfg.get("Dataset", {}))
    # Main configs intentionally leave environment-specific dataset paths empty.
    dataset_cfg["dataset_root"] = args.dataset_root
    dataset_cfg["eval_videos"] = _parse_video_filter(args.videos)
    cfg["Dataset"] = dataset_cfg

    decoder_cfg = dict(cfg.get("Decoder", {}))
    decoder_cfg["num_queries"] = int(num_queries)
    decoder_cfg["hqg_num_learnable_memory"] = int(hqg_memory)
    cfg["Decoder"] = decoder_cfg

    visualizer_cfg = dict(cfg.get("Visualizer", {}))
    visualizer_cfg["enabled"] = True
    visualizer_cfg["output_root_dir"] = str(run_root)
    visualizer_cfg = _configure_focus_task(visualizer_cfg=visualizer_cfg, compose_header_mode=str(args.compose_header_mode))
    cfg["Visualizer"] = visualizer_cfg

    train_cfg = _try_load_checkpoint_train_config(str(checkpoint_path.parent))
    if train_cfg is not None:
        _merge_eval_with_train_model_config(cfg, train_cfg, str(checkpoint_path.parent / "train" / "config.yaml"))
    else:
        missing_keys = _missing_model_config_keys(cfg)
        if missing_keys:
            raise KeyError(
                f"Missing eval model config keys: {missing_keys}. "
                f"Either restore checkpoint train/config.yaml under {checkpoint_path.parent} or add them to the eval yaml."
            )

    return cfg


def _resolve_visualize_root(cfg: dict, sub_out_path: str) -> Path:
    inference_root = Path(_resolve_inference_root(cfg=cfg, mode=str(cfg["MODE"]))).resolve()
    visualizer_cfg = dict(cfg.get("Visualizer", {}))
    root_dir_name = str(visualizer_cfg.get("root_dir_name", f"visualize_{sub_out_path}"))
    output_root_dir = str(visualizer_cfg.get("output_root_dir", "")).strip()
    if output_root_dir:
        base_root = Path(output_root_dir).expanduser()
        if not base_root.is_absolute():
            base_root = inference_root / base_root
    else:
        base_root = inference_root
    return (base_root / root_dir_name).resolve()


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _task_dir(root_dir: Path, video_name: str, task_name: str) -> Path:
    return root_dir / video_name / task_name


def _find_common_videos(method_roots: list[Path], task_name: str, video_filter: list[str] | None) -> list[str]:
    if video_filter is not None:
        return [video for video in video_filter if all(_task_dir(root, video, task_name).is_dir() for root in method_roots)]

    common = None
    for root in method_roots:
        videos = {path.name for path in root.iterdir() if path.is_dir() and _task_dir(root, path.name, task_name).is_dir()}
        common = videos if common is None else (common & videos)
    return sorted(common or [])


def _find_common_stems(method_roots: list[Path], video_name: str, task_name: str) -> list[str]:
    common = None
    for root in method_roots:
        summary = _load_method_bundle_summary(_task_dir(root, video_name, task_name) / BUNDLE_FILENAME)
        stems = {str(frame["stem"]) for frame in list(summary.get("frames", []))}
        common = stems if common is None else (common & stems)
    return sorted(common or [])


def _load_meta(meta_path: Path) -> dict[str, np.ndarray | str]:
    with np.load(meta_path, allow_pickle=False) as data:
        meta = {key: data[key] for key in data.files}
    meta["image_path"] = str(np.atleast_1d(meta["image_path"]).item())
    return meta


def _scalar_from_meta(meta: dict[str, np.ndarray | str], key: str, default: int = 0) -> int:
    value = meta.get(key, None)
    if value is None:
        return int(default)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return int(default)
        return int(np.atleast_1d(value).item())
    return int(value)


def _array_from_meta(meta: dict[str, np.ndarray | str], key: str) -> list[list[float]]:
    value = meta.get(key, None)
    if not isinstance(value, np.ndarray) or value.size == 0:
        return []
    return np.asarray(value, dtype=np.float32).tolist()


def _build_video_summary(task_dir: Path, task_name: str, overwrite: bool = True) -> dict:
    del task_name, overwrite
    return _load_method_bundle_summary(task_dir / BUNDLE_FILENAME)


def _load_method_bundle_summary(bundle_path: Path) -> dict:
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Query-focus bundle not found: {bundle_path}")
    with zipfile.ZipFile(bundle_path, "r") as zf:
        with zf.open(BUNDLE_SUMMARY_FILENAME, "r") as f:
            return json.loads(f.read().decode("utf-8"))


def _build_all_video_summaries(method_roots: list[Path], task_name: str, videos: list[str] | None) -> dict[str, dict[str, dict]]:
    summaries: dict[str, dict[str, dict]] = {}
    for root_dir in method_roots:
        root_key = str(root_dir)
        summaries[root_key] = {}
        available_videos = videos
        if available_videos is None:
            available_videos = sorted(path.name for path in root_dir.iterdir() if path.is_dir())
        for video_name in available_videos:
            task_dir = _task_dir(root_dir, video_name, task_name)
            if not task_dir.is_dir():
                continue
            summaries[root_key][video_name] = _build_video_summary(task_dir=task_dir, task_name=task_name, overwrite=True)
    return summaries


def _build_frame_lookup(summary: dict) -> dict[str, dict]:
    return {str(frame["stem"]): frame for frame in list(summary.get("frames", []))}


def _draw_title_bar(image: np.ndarray, title: str) -> np.ndarray:
    bar_h = 38
    out = np.zeros((image.shape[0] + bar_h, image.shape[1], 3), dtype=np.uint8)
    out[:bar_h] = (22, 22, 22)
    out[bar_h:] = image
    cv2.putText(out, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def _load_panel_image(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return img


def _iter_with_progress(items: Iterable[_T], total: int, desc: str) -> Iterator[_T]:
    total = max(0, int(total))
    if total <= 0:
        return iter(())

    width = 24

    def _render(index: int):
        filled = int(round(width * (index / max(total, 1))))
        bar = "#" * filled + "-" * max(0, width - filled)
        print(f"\r[QueryFocus] {desc}: [{bar}] {index}/{total}", end="", flush=True)

    def _generator() -> Iterator[_T]:
        _render(0)
        for index, item in enumerate(items, start=1):
            yield item
            _render(index)
        print("", flush=True)

    return _generator()


def _clip_crop_bounds(start: float, size: float, limit: int) -> tuple[int, int]:
    size = min(float(limit), max(1.0, float(size)))
    start = float(start)
    end = start + size
    if start < 0.0:
        end -= start
        start = 0.0
    if end > float(limit):
        start -= end - float(limit)
        end = float(limit)
    start = max(0.0, start)
    end = min(float(limit), end)
    if end - start < 1.0:
        end = min(float(limit), start + 1.0)
    return int(np.floor(start)), int(np.ceil(end))


def _compute_center_crop_box(image_shape: tuple[int, int], gt_boxes: np.ndarray, crop_margin: int) -> tuple[int, int, int, int]:
    img_h, img_w = image_shape[:2]
    if gt_boxes.size == 0:
        return 0, 0, img_w, img_h

    x1 = float(np.min(gt_boxes[:, 0])) - float(crop_margin)
    y1 = float(np.min(gt_boxes[:, 1])) - float(crop_margin)
    x2 = float(np.max(gt_boxes[:, 2])) + float(crop_margin)
    y2 = float(np.max(gt_boxes[:, 3])) + float(crop_margin)

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(img_w), x2)
    y2 = min(float(img_h), y2)

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    aspect = float(img_w) / max(float(img_h), 1.0)

    crop_w = max(box_w, box_h * aspect)
    crop_h = crop_w / max(aspect, 1e-6)
    if crop_h < box_h:
        crop_h = box_h
        crop_w = crop_h * aspect

    crop_x1, crop_x2 = _clip_crop_bounds(cx - 0.5 * crop_w, crop_w, img_w)
    crop_y1, crop_y2 = _clip_crop_bounds(cy - 0.5 * crop_h, crop_h, img_h)
    return crop_x1, crop_y1, crop_x2, crop_y2


def _crop_image(image: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    return image[y1:y2, x1:x2]


def _resize_panel(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    if image.shape[1] == target_w and image.shape[0] == target_h:
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _build_original_header(frame_summary: dict) -> str:
    return f"Original | Frame {int(frame_summary['frame_id'])} | gt {int(frame_summary['all_gt_count'])}"


def _build_method_header(label: str, frame_summary: dict) -> str:
    return (
        f"{label} | Frame {int(frame_summary['frame_id'])} | "
        f"det {int(frame_summary['num_valid_queries'])} | "
        f"prev {int(frame_summary['prev_track_count'])} | "
        f"focus {int(frame_summary['focus_gt_count'])}"
    )


def _build_frame_crop_map(
    first_frame_lookup: dict[str, dict],
    image_shape: tuple[int, int],
    crop_scope: str,
    crop_margin: int,
) -> tuple[dict[str, tuple[int, int, int, int]], tuple[int, int]]:
    img_h, img_w = image_shape[:2]
    full_box = (0, 0, img_w, img_h)
    if crop_scope == "none":
        return {stem: full_box for stem in first_frame_lookup.keys()}, (img_w, img_h)

    if crop_scope == "video":
        all_boxes = []
        for frame_summary in first_frame_lookup.values():
            gt_boxes = np.asarray(frame_summary.get("all_gt_boxes", []), dtype=np.float32)
            if gt_boxes.size > 0:
                all_boxes.append(gt_boxes)
        merged_boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4), dtype=np.float32)
        crop_box = _compute_center_crop_box(image_shape=image_shape, gt_boxes=merged_boxes, crop_margin=int(crop_margin))
        crop_map = {stem: crop_box for stem in first_frame_lookup.keys()}
        return crop_map, (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])

    crop_map: dict[str, tuple[int, int, int, int]] = {}
    max_w, max_h = 1, 1
    for stem, frame_summary in first_frame_lookup.items():
        gt_boxes = np.asarray(frame_summary.get("all_gt_boxes", []), dtype=np.float32)
        crop_box = _compute_center_crop_box(image_shape=image_shape, gt_boxes=gt_boxes, crop_margin=int(crop_margin))
        crop_map[stem] = crop_box
        max_w = max(max_w, int(crop_box[2] - crop_box[0]))
        max_h = max(max_h, int(crop_box[3] - crop_box[1]))
    return crop_map, (max_w, max_h)


def _compose_panels(
    method_roots: list[Path],
    labels: list[str],
    summaries: dict[str, dict[str, dict]],
    output_dir: Path,
    task_name: str,
    videos: list[str] | None,
    overwrite: bool,
    include_original: bool,
    crop_scope: str,
    crop_margin: int,
    compose_header_mode: str,
):
    output_dir = _prepare_output_dir(output_dir=output_dir, overwrite=overwrite)
    common_videos = _find_common_videos(method_roots=method_roots, task_name=task_name, video_filter=videos)
    if not common_videos:
        raise RuntimeError("No common videos found across the provided method outputs.")

    for video_name in _iter_with_progress(common_videos, total=len(common_videos), desc="compose videos"):
        frame_stems = _find_common_stems(method_roots=method_roots, video_name=video_name, task_name=task_name)
        if not frame_stems:
            continue

        video_out_dir = output_dir / video_name
        video_out_dir.mkdir(parents=True, exist_ok=True)

        frame_lookups = {
            str(root_dir): _build_frame_lookup(summaries[str(root_dir)][video_name]) for root_dir in method_roots
        }
        first_lookup = frame_lookups[str(method_roots[0])]
        first_frame_summary = next(iter(first_lookup.values()))
        first_frame_image = _load_panel_image(Path(str(first_frame_summary["image_path"])))
        crop_map, target_panel_size = _build_frame_crop_map(
            first_frame_lookup=first_lookup,
            image_shape=first_frame_image.shape[:2],
            crop_scope=str(crop_scope),
            crop_margin=int(crop_margin),
        )
        compose_summary = {
            "video_name": video_name,
            "task_name": task_name,
            "include_original": bool(include_original),
            "compose_header_mode": str(compose_header_mode),
            "crop_scope": str(crop_scope),
            "crop_margin": int(crop_margin),
            "target_panel_size": [int(target_panel_size[0]), int(target_panel_size[1])],
            "methods": [
                {"label": str(label), "root_dir": str(root_dir)} for root_dir, label in zip(method_roots, labels)
            ],
            "frames": [],
        }

        for stem in _iter_with_progress(frame_stems, total=len(frame_stems), desc=f"compose {video_name}"):
            first_summary = frame_lookups[str(method_roots[0])][stem]
            orig_img = _load_panel_image(Path(str(first_summary["image_path"])))
            crop_box = crop_map[stem]

            panels = []
            if include_original:
                orig_panel = _crop_image(orig_img, crop_box) if str(crop_scope) != "none" else orig_img
                orig_panel = _resize_panel(orig_panel, target_panel_size)
                if compose_header_mode == "compose":
                    orig_panel = _draw_title_bar(orig_panel, _build_original_header(first_summary))
                elif compose_header_mode == "task":
                    orig_panel = _draw_title_bar(orig_panel, "Original")
                panels.append(orig_panel)

            method_frame_summaries = {}
            for root_dir, label in zip(method_roots, labels):
                render_path = _task_dir(root_dir, video_name, task_name) / f"{stem}.jpg"
                render_img = _load_panel_image(render_path)
                render_panel = _crop_image(render_img, crop_box) if str(crop_scope) != "none" else render_img
                render_panel = _resize_panel(render_panel, target_panel_size)
                frame_summary = frame_lookups[str(root_dir)][stem]
                if compose_header_mode == "compose":
                    render_panel = _draw_title_bar(
                        render_panel,
                        _build_method_header(label=label, frame_summary=frame_summary),
                    )
                elif compose_header_mode == "task":
                    render_panel = _draw_title_bar(render_panel, str(label))
                panels.append(render_panel)
                method_frame_summaries[str(label)] = {
                    "num_valid_queries": int(frame_summary["num_valid_queries"]),
                    "prev_track_count": int(frame_summary["prev_track_count"]),
                    "focus_gt_count": int(frame_summary["focus_gt_count"]),
                    "all_gt_count": int(frame_summary["all_gt_count"]),
                    "heat_stats": dict(frame_summary.get("heat_stats", {})),
                }

            concat = np.concatenate(panels, axis=1)
            cv2.imwrite(str(video_out_dir / f"{stem}.jpg"), concat)
            compose_summary["frames"].append(
                {
                    "stem": str(stem),
                    "frame_id": int(first_summary["frame_id"]),
                    "image_path": str(first_summary["image_path"]),
                    "crop_box": [int(v) for v in crop_box],
                    "panel_size": [int(target_panel_size[0]), int(target_panel_size[1])],
                    "original": {
                        "all_gt_count": int(first_summary["all_gt_count"]),
                        "focus_gt_count": int(first_summary["focus_gt_count"]),
                    },
                    "methods": method_frame_summaries,
                }
            )

        with open(video_out_dir / SUMMARY_FILENAME, "w", encoding="utf-8") as f:
            json.dump(compose_summary, f, ensure_ascii=False)


def _build_compose_output_dir(args: argparse.Namespace) -> Path:
    run_root = _build_run_root(args)
    videos = _parse_video_filter(args.videos)
    video_tag = _build_video_tag(videos=videos, mode=str(args.mode))
    return run_root / f"query_focus_compose_{video_tag}"


def _run_inference_and_collect_method_roots(args: argparse.Namespace, base_cfg: dict) -> list[Path]:
    run_root = _build_run_root(args)
    method_roots: list[Path] = []
    for ckpt_dir, label, num_queries, hqg_memory in zip(
        args.method_ckpt_dirs,
        args.method_labels,
        args.method_num_queries,
        args.method_hqg_memories,
    ):
        checkpoint_path = _resolve_checkpoint_path(ckpt_dir)
        runtime_cfg = _prepare_runtime_config(
            base_cfg=base_cfg,
            args=args,
            run_root=run_root,
            label=label,
            checkpoint_path=checkpoint_path,
            num_queries=int(num_queries),
            hqg_memory=int(hqg_memory),
        )
        sub_out_path = _build_eval_sub_out_path(cfg=runtime_cfg, eval_file_path=str(checkpoint_path))
        runtime_cfg.setdefault("Visualizer", {})
        runtime_cfg["Visualizer"]["root_dir_name"] = f"visualize_{sub_out_path}"
        method_roots.append(_resolve_visualize_root(cfg=runtime_cfg, sub_out_path=sub_out_path))
        print(f"[QueryFocus] running {label} from {checkpoint_path}")
        inference_offline(config=runtime_cfg)
    return method_roots


def _collect_existing_method_roots(args: argparse.Namespace, base_cfg: dict) -> list[Path]:
    run_root = _build_run_root(args)
    method_roots: list[Path] = []
    for ckpt_dir, label, num_queries, hqg_memory in zip(
        args.method_ckpt_dirs,
        args.method_labels,
        args.method_num_queries,
        args.method_hqg_memories,
    ):
        checkpoint_path = _resolve_checkpoint_path(ckpt_dir)
        runtime_cfg = _prepare_runtime_config(
            base_cfg=base_cfg,
            args=args,
            run_root=run_root,
            label=label,
            checkpoint_path=checkpoint_path,
            num_queries=int(num_queries),
            hqg_memory=int(hqg_memory),
        )
        sub_out_path = _build_eval_sub_out_path(cfg=runtime_cfg, eval_file_path=str(checkpoint_path))
        runtime_cfg.setdefault("Visualizer", {})
        runtime_cfg["Visualizer"]["root_dir_name"] = f"visualize_{sub_out_path}"
        method_roots.append(_resolve_visualize_root(cfg=runtime_cfg, sub_out_path=sub_out_path))
    return method_roots


def main() -> None:
    args = parse_args()
    base_cfg = _load_yaml_config(args.config_path)
    run_root = _build_run_root(args)

    # Mirror main.py runtime setup so direct script entry stays behaviorally aligned with standard eval runs.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(base_cfg["Training"]["Available_gpus"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if bool(args.compose_only):
        method_roots = _collect_existing_method_roots(args=args, base_cfg=base_cfg)
    else:
        method_roots = _run_inference_and_collect_method_roots(args=args, base_cfg=base_cfg)

    if bool(args.skip_compose):
        print("[QueryFocus] skip compose as requested.")
        print(f"[QueryFocus] run root: {run_root}")
        for label, root_dir in zip(args.method_labels, method_roots):
            print(f"[QueryFocus] {label}: {root_dir}")
        return

    summaries = _build_all_video_summaries(
        method_roots=method_roots,
        task_name=str(args.task_name),
        videos=_parse_video_filter(args.videos),
    )
    compose_output_dir = _build_compose_output_dir(args=args)
    _compose_panels(
        method_roots=method_roots,
        labels=[str(label) for label in args.method_labels],
        summaries=summaries,
        output_dir=compose_output_dir,
        task_name=str(args.task_name),
        videos=_parse_video_filter(args.videos),
        overwrite=bool(args.overwrite),
        include_original=bool(args.include_original),
        crop_scope=str(args.crop_scope),
        crop_margin=int(args.crop_margin),
        compose_header_mode=str(args.compose_header_mode),
    )
    print("[QueryFocus] method outputs:")
    print(f"[QueryFocus] run root: {run_root}")
    for label, root_dir in zip(args.method_labels, method_roots):
        print(f"  {label}: {root_dir}")
    print(f"[QueryFocus] composed panels: {compose_output_dir}")


if __name__ == "__main__":
    main()
