from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None

from ...TrackEval.trackeval.datasets._base_dataset import _BaseDataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
STATUS_MATCH = "match"
STATUS_IDSW = "idsw"
STATUS_LOST = "lost"
STATUS_ABSENT = "absent"
STATUS_LABELS = {
    STATUS_MATCH: "M",
    STATUS_IDSW: "S",
    STATUS_LOST: "L",
    STATUS_ABSENT: "-",
}
STATUS_COLORS = {
    STATUS_MATCH: (60, 220, 60),
    STATUS_IDSW: (40, 220, 240),
    STATUS_LOST: (50, 70, 240),
    STATUS_ABSENT: (170, 170, 170),
}
GT_COLOR_BGR = (30, 255, 30)
METHOD_COLORS_BGR = (
    (0, 235, 255),    # yellow-ish
    (35, 35, 255),    # red-ish
    (255, 130, 30),   # blue-ish
)


@dataclass(slots=True)
class TargetMatch:
    gt_id: int
    gt_box_xywh: tuple[float, float, float, float]
    status: str
    pred_id: int | None
    pred_box_xywh: tuple[float, float, float, float] | None
    iou: float


@dataclass(slots=True)
class KeyEvent:
    video_name: str
    frame_id: int
    gt_id: int
    primary_status: str
    trigger_methods: list[str]


@dataclass(slots=True)
class VideoRenderPlan:
    video_name: str
    frame_infos: list[tuple[int, Path]]
    matches_by_tracker: list[dict[int, dict[int, TargetMatch]]]
    tracker_results_by_method: list[dict[int, np.ndarray]]
    render_targets_by_frame: dict[int, set[int]]
    key_events: list[KeyEvent]
    summary_rows: list[dict[str, str | int | float]]
    planned_frames: int


@dataclass(slots=True)
class StableClipCandidate:
    video_name: str
    gt_id: int
    start_frame: int
    end_frame: int
    primary_pred_id: int
    key_frames: tuple[int, ...]
    related_pred_ids_by_method: dict[int, tuple[int, ...]]
    switched_methods: tuple[str, ...]


class MultiTrackerCompareTask:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.compare = bool(self.cfg.get("compare", True))
        self.iou_threshold = float(self.cfg.get("iou_threshold", 0.5))
        self.context_frames = max(0, int(self.cfg.get("context_frames", 5)))
        self.primary_tracker_index = int(self.cfg.get("primary_tracker_index", -1))
        self.primary_require_match = bool(self.cfg.get("primary_require_match", True))
        self.compare_statuses = {
            str(status).strip().lower()
            for status in self.cfg.get("compare_statuses", [STATUS_IDSW, STATUS_LOST])
            if str(status).strip()
        }
        self.search_stable_clip = bool(self.cfg.get("search_stable_clip", False))
        self.search_window = max(2, int(self.cfg.get("search_window", 20)))
        raw_search_limit = int(self.cfg.get("search_limit", -1))
        self.search_limit = -1 if raw_search_limit <= 0 else raw_search_limit
        self.output_dir = Path(str(self.cfg.get("output_dir", ""))).expanduser().resolve()
        self.overwrite = bool(self.cfg.get("overwrite", False))
        self.border_alpha = float(np.clip(float(self.cfg.get("border_alpha", 0.5)), 0.0, 1.0))
        self.line_thickness = max(1, int(self.cfg.get("line_thickness", 2)))
        self.label_font_scale = float(self.cfg.get("label_font_scale", 0.45))
        self.max_table_rows = max(1, int(self.cfg.get("max_table_rows", 8)))
        self.max_frames = max(0, int(self.cfg.get("max_frames", 0)))
        self.render_non_key_targets = bool(self.cfg.get("render_non_key_targets", False))
        self.target_frames_by_video = self._normalize_target_frames(self.cfg.get("target_frames_by_video", {}))
        self.anchor_text_step = max(12, int(self.cfg.get("anchor_text_step", 16)))
        self.show_related_idsw_boxes = bool(self.cfg.get("show_related_idsw_boxes", self.search_stable_clip))
        self.related_box_mode = str(self.cfg.get("related_box_mode", "idsw")).strip().lower()
        raw_focus_gt_ids = self.cfg.get("focus_gt_ids", self.cfg.get("focus_gt_id", None))
        if raw_focus_gt_ids in (None, "", "None"):
            self.focus_gt_ids = tuple()
        elif isinstance(raw_focus_gt_ids, (list, tuple, set)):
            self.focus_gt_ids = tuple(int(gt_id) for gt_id in raw_focus_gt_ids)
        else:
            self.focus_gt_ids = tuple(int(token.strip()) for token in str(raw_focus_gt_ids).split(",") if token.strip())
        self.tracker_dirs = [Path(path).expanduser().resolve() for path in self.cfg.get("tracker_dirs", [])]
        self.tracker_labels = [str(label).strip() for label in self.cfg.get("tracker_labels", [])]
        self.selected_clip_candidates: list[StableClipCandidate] = []

    def run(self, dataset) -> bool:
        if not self._validate_task_config():
            return False

        prepared = self._prepare_runtime(dataset=dataset)
        if prepared is None:
            return False

        selected_videos, frames_by_video, tracker_results = prepared
        key_events_all: list[KeyEvent] = []
        summary_rows: list[dict[str, str | int | float]] = []
        rendered_videos = 0
        rendered_frames = 0
        video_plans: list[VideoRenderPlan] = []

        for video_name in selected_videos:
            if video_name not in tracker_results:
                self._warn(f"video={video_name} has no valid tracker results after validation, skipped.")
                continue

            frame_infos = frames_by_video.get(video_name, [])
            if not frame_infos:
                self._warn(f"video={video_name} has no image frames, skipped.")
                continue

            matches_by_tracker = self._build_video_matches(
                video_name=video_name,
                frame_infos=frame_infos,
                gt_frames=dataset.gts.get(video_name, {}),
                tracker_results_by_method=tracker_results[video_name],
            )
            render_targets_by_frame, key_events = self._select_render_targets(
                video_name=video_name,
                frame_infos=frame_infos,
                matches_by_tracker=matches_by_tracker,
            )
            video_summary_rows = self._build_summary_rows(video_name=video_name, matches_by_tracker=matches_by_tracker)
            planned_frames = self._count_planned_frames(
                video_name=video_name,
                frame_infos=frame_infos,
                render_targets_by_frame=render_targets_by_frame,
            )
            key_events_all.extend(key_events)
            summary_rows.extend(video_summary_rows)
            video_plans.append(
                VideoRenderPlan(
                    video_name=video_name,
                    frame_infos=frame_infos,
                    matches_by_tracker=matches_by_tracker,
                    tracker_results_by_method=tracker_results[video_name],
                    render_targets_by_frame=render_targets_by_frame,
                    key_events=key_events,
                    summary_rows=video_summary_rows,
                    planned_frames=planned_frames,
                )
            )

        if self.search_stable_clip:
            self.selected_clip_candidates = self._search_stable_clips(video_plans=video_plans)
            self._apply_search_candidates(video_plans=video_plans, candidates=self.selected_clip_candidates)
        total_planned_frames = sum(
            self._count_planned_frames(
                video_name=video_plan.video_name,
                frame_infos=video_plan.frame_infos,
                render_targets_by_frame=video_plan.render_targets_by_frame,
            )
            for video_plan in video_plans
        )

        progress = self._build_progress(total_frames=total_planned_frames, mode=getattr(dataset, "mode", ""))
        for video_plan in video_plans:
            num_rendered = self._render_video(
                video_name=video_plan.video_name,
                frame_infos=video_plan.frame_infos,
                matches_by_tracker=video_plan.matches_by_tracker,
                tracker_results_by_method=video_plan.tracker_results_by_method,
                render_targets_by_frame=video_plan.render_targets_by_frame,
                progress=progress,
            )
            if num_rendered <= 0:
                continue
            rendered_videos += 1
            rendered_frames += num_rendered
        if progress is not None:
            progress.close()

        self._write_summary(
            selected_videos=selected_videos,
            rendered_videos=rendered_videos,
            rendered_frames=rendered_frames,
            key_events=key_events_all,
            summary_rows=summary_rows,
        )
        print(f"Output dir: {self.output_dir}")
        print(f"Selected videos ({len(selected_videos)}): {', '.join(selected_videos)}")
        print(f"Rendered videos: {rendered_videos}")
        print(f"Rendered frames: {rendered_frames}")
        print(f"Compare mode: {self.compare}")
        print(f"Focus GT IDs: {list(self.focus_gt_ids) if self.focus_gt_ids else None}")
        print(f"Target frames: {self._format_target_frames() if self.target_frames_by_video else None}")
        print(f"Search stable clip: {self.search_stable_clip}")
        print(f"Stable clip candidates: {len(self.selected_clip_candidates)}")
        print(f"Key events: {len(key_events_all)}")
        return rendered_videos > 0

    @staticmethod
    def _normalize_target_frames(raw_target_frames) -> dict[str, set[int]]:
        if not raw_target_frames:
            return {}
        if not isinstance(raw_target_frames, dict):
            raise TypeError(f"target_frames_by_video must be a dict, got: {type(raw_target_frames)}")
        normalized: dict[str, set[int]] = {}
        for video_name, frame_ids in raw_target_frames.items():
            video_key = str(video_name).strip()
            if not video_key:
                continue
            frame_set = {int(frame_id) for frame_id in frame_ids}
            if any(frame_id <= 0 for frame_id in frame_set):
                raise ValueError(f"target frame ids must be positive for video={video_key}: {sorted(frame_set)}")
            normalized[video_key] = frame_set
        return normalized

    def _target_frame_filter(self, video_name: str) -> set[int] | None:
        if not self.target_frames_by_video:
            return None
        return self.target_frames_by_video.get(video_name, set())

    def _format_target_frames(self) -> str:
        return ";".join(
            f"{video_name}:{','.join(str(frame_id) for frame_id in sorted(frame_ids))}"
            for video_name, frame_ids in sorted(self.target_frames_by_video.items())
        )


    def _validate_task_config(self) -> bool:
        if len(self.tracker_dirs) == 0 or len(self.tracker_labels) == 0:
            self._warn("tracker_dirs / tracker_labels is empty, task skipped.")
            return False
        if len(self.tracker_dirs) != len(self.tracker_labels):
            self._warn("tracker_dirs and tracker_labels length mismatch, task skipped.")
            return False
        if len(set(self.tracker_labels)) != len(self.tracker_labels):
            self._warn(f"tracker_labels must be unique: {self.tracker_labels}, task skipped.")
            return False
        if any(gt_id <= 0 for gt_id in self.focus_gt_ids):
            self._warn(f"focus_gt_ids must be positive, got {list(self.focus_gt_ids)}, task skipped.")
            return False
        if self.related_box_mode not in {"idsw"}:
            self._warn(f"unsupported related_box_mode={self.related_box_mode}, task skipped.")
            return False
        if self.target_frames_by_video and self.search_stable_clip:
            self._warn("target_frames_by_video is not supported with search_stable_clip, task skipped.")
            return False
        for tracker_dir in self.tracker_dirs:
            if not tracker_dir.is_dir():
                self._warn(f"tracker_dir does not exist: {tracker_dir}, task skipped.")
                return False
        if self.output_dir in (Path(""), Path(".")):
            self._warn("output_dir is empty, task skipped.")
            return False
        if self.output_dir.exists():
            if self.overwrite:
                shutil.rmtree(self.output_dir)
            else:
                self._warn(f"output_dir already exists: {self.output_dir}, task skipped.")
                return False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True

    def _prepare_runtime(self, dataset) -> tuple[list[str], dict[str, list[tuple[int, Path]]], dict[str, list[dict[int, np.ndarray]]]] | None:
        try:
            selected_videos = dataset.get_selected_videos(mode=dataset.mode)
        except Exception as exc:
            self._warn(f"dataset selected_videos resolve failed: {exc}, task skipped.")
            return None
        if not selected_videos:
            self._warn("selected_videos is empty, task skipped.")
            return None

        frames_by_video: dict[str, list[tuple[int, Path]]] = {}
        tracker_results: dict[str, list[dict[int, np.ndarray]]] = {}
        valid_videos: list[str] = []
        for video_name in selected_videos:
            frame_infos = self._iter_video_frames(Path(dataset.split_dir) / video_name)
            if frame_infos is None:
                self._warn(f"video={video_name} image frames invalid, skipped.")
                continue
            gt_frames = dataset.gts.get(video_name, None)
            if gt_frames is None or len(gt_frames) == 0:
                self._warn(f"video={video_name} gt missing, skipped.")
                continue

            video_tracker_results: list[dict[int, np.ndarray]] = []
            skip_video = False
            image_frame_ids = {frame_id for frame_id, _ in frame_infos}
            for tracker_label, tracker_dir in zip(self.tracker_labels, self.tracker_dirs):
                result_path = tracker_dir / f"{video_name}.txt"
                loaded = self._load_tracker_results(result_path=result_path, video_name=video_name, tracker_label=tracker_label)
                if loaded is None:
                    skip_video = True
                    break
                extra_frames = sorted(frame_id for frame_id in loaded.keys() if frame_id not in image_frame_ids)
                if extra_frames:
                    self._warn(
                        f"video={video_name} tracker={tracker_label} has frame ids outside image range: "
                        f"{extra_frames[:5]}{'...' if len(extra_frames) > 5 else ''}, ignored."
                    )
                    loaded = {frame_id: rows for frame_id, rows in loaded.items() if frame_id in image_frame_ids}
                video_tracker_results.append(loaded)
            if skip_video:
                continue

            frames_by_video[video_name] = frame_infos
            tracker_results[video_name] = video_tracker_results
            valid_videos.append(video_name)

        if not valid_videos:
            self._warn("no valid videos left after validation, task skipped.")
            return None
        return valid_videos, frames_by_video, tracker_results

    def _build_progress(self, total_frames: int, mode: str):
        if total_frames <= 0:
            return None
        if tqdm is not None:
            return tqdm(
                total=total_frames,
                desc=f"Offline compare render {mode}",
                unit="it",
                dynamic_ncols=True,
            )
        print(f"Offline compare render {mode}: 0/{total_frames}")
        return None

    def _count_planned_frames(
        self,
        video_name: str,
        frame_infos: list[tuple[int, Path]],
        render_targets_by_frame: dict[int, set[int]],
    ) -> int:
        target_frames = self._target_frame_filter(video_name)

        def allowed(frame_id: int) -> bool:
            return target_frames is None or frame_id in target_frames

        planned = 0
        for frame_id, _ in frame_infos:
            if not allowed(frame_id):
                continue
            if self.search_stable_clip:
                if frame_id not in render_targets_by_frame:
                    continue
            elif not self.focus_gt_ids and self.compare and frame_id not in render_targets_by_frame:
                continue
            planned += 1
            if self.max_frames > 0 and planned >= self.max_frames:
                break
        return planned

    @staticmethod
    def _iter_video_frames(video_dir: Path) -> list[tuple[int, Path]] | None:
        img_dir = video_dir / "img1"
        if not img_dir.is_dir():
            return None
        frame_infos: list[tuple[int, Path]] = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                frame_id = int(img_path.stem)
            except ValueError:
                continue
            frame_infos.append((frame_id, img_path))
        return frame_infos or None

    def _load_tracker_results(self, result_path: Path, video_name: str, tracker_label: str) -> dict[int, np.ndarray] | None:
        if not result_path.is_file():
            self._warn(f"video={video_name} tracker={tracker_label} result missing: {result_path}")
            return None
        if result_path.stat().st_size == 0:
            self._warn(f"video={video_name} tracker={tracker_label} result empty: {result_path}")
            return {}
        try:
            result = np.loadtxt(result_path, delimiter=",", ndmin=2, dtype=np.float32)
        except Exception as exc:
            self._warn(f"video={video_name} tracker={tracker_label} result load failed: {exc}")
            return None
        if result.size == 0:
            return {}
        if result.shape[1] < 6:
            self._warn(f"video={video_name} tracker={tracker_label} result has fewer than 6 columns, skipped.")
            return None

        frame_ids = result[:, 0].astype(np.int32)
        track_ids = result[:, 1].astype(np.int64)
        grouped: dict[int, np.ndarray] = {}
        for frame_id in np.unique(frame_ids):
            rows = np.ascontiguousarray(result[frame_ids == frame_id])
            ids_t = rows[:, 1].astype(np.int64)
            if len(np.unique(ids_t)) != len(ids_t):
                self._warn(
                    f"video={video_name} tracker={tracker_label} frame={int(frame_id)} has duplicate tracker ids, skipped."
                )
                return None
            grouped[int(frame_id)] = rows
        if np.any(track_ids < 0):
            self._warn(f"video={video_name} tracker={tracker_label} contains negative tracker ids, kept as-is.")
        return grouped

    def _build_video_matches(
        self,
        video_name: str,
        frame_infos: list[tuple[int, Path]],
        gt_frames: dict[int, list[list[float]]],
        tracker_results_by_method: list[dict[int, np.ndarray]],
    ) -> list[dict[int, dict[int, TargetMatch]]]:
        del video_name
        matches_by_tracker: list[dict[int, dict[int, TargetMatch]]] = []
        for tracker_results in tracker_results_by_method:
            matches_by_tracker.append(self._match_one_tracker(frame_infos=frame_infos, gt_frames=gt_frames, tracker_results=tracker_results))
        return matches_by_tracker

    def _match_one_tracker(
        self,
        frame_infos: list[tuple[int, Path]],
        gt_frames: dict[int, list[list[float]]],
        tracker_results: dict[int, np.ndarray],
    ) -> dict[int, dict[int, TargetMatch]]:
        frame_matches: dict[int, dict[int, TargetMatch]] = {}
        prev_tracker_id: dict[int, int] = {}
        prev_timestep_tracker_id: dict[int, int] = {}
        eps = np.finfo("float").eps

        for frame_id, _ in frame_infos:
            gt_entries = [
                gt for gt in gt_frames.get(frame_id, [])
                if len(gt) >= 6 and float(gt[5]) > 0.0
            ]
            tracker_rows = tracker_results.get(frame_id, np.empty((0, 9), dtype=np.float32))
            gt_ids = np.asarray([int(gt[0]) for gt in gt_entries], dtype=np.int64)
            gt_boxes = np.asarray([gt[1:5] for gt in gt_entries], dtype=np.float32) if gt_entries else np.empty((0, 4), dtype=np.float32)
            tracker_ids = tracker_rows[:, 1].astype(np.int64) if tracker_rows.size > 0 else np.empty((0,), dtype=np.int64)
            tracker_boxes = tracker_rows[:, 2:6].astype(np.float32) if tracker_rows.size > 0 else np.empty((0, 4), dtype=np.float32)

            matches_t: dict[int, TargetMatch] = {}
            if len(gt_ids) == 0:
                frame_matches[frame_id] = matches_t
                prev_timestep_tracker_id = {}
                continue

            if len(tracker_ids) == 0:
                for gt_id, gt_box in zip(gt_ids.tolist(), gt_boxes.tolist()):
                    matches_t[gt_id] = TargetMatch(
                        gt_id=gt_id,
                        gt_box_xywh=tuple(float(v) for v in gt_box),
                        status=STATUS_LOST,
                        pred_id=None,
                        pred_box_xywh=None,
                        iou=0.0,
                    )
                frame_matches[frame_id] = matches_t
                prev_timestep_tracker_id = {}
                continue

            similarity = _BaseDataset._calculate_box_ious(gt_boxes, tracker_boxes, box_format="xywh")
            score_mat = (tracker_ids[np.newaxis, :] == np.asarray(
                [prev_timestep_tracker_id.get(int(gt_id), -10**12) for gt_id in gt_ids],
                dtype=np.int64
            )[:, np.newaxis]).astype(np.float32)
            score_mat = 1000.0 * score_mat + similarity
            score_mat[similarity < self.iou_threshold - eps] = 0.0
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched_mask = score_mat[match_rows, match_cols] > 0.0 + eps
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            matched_gt_ids = set()
            new_prev_timestep: dict[int, int] = {}
            for row_idx, col_idx in zip(match_rows.tolist(), match_cols.tolist()):
                gt_id = int(gt_ids[row_idx])
                pred_id = int(tracker_ids[col_idx])
                gt_box = tuple(float(v) for v in gt_boxes[row_idx].tolist())
                pred_box = tuple(float(v) for v in tracker_boxes[col_idx].tolist())
                is_idsw = gt_id in prev_tracker_id and prev_tracker_id[gt_id] != pred_id
                matches_t[gt_id] = TargetMatch(
                    gt_id=gt_id,
                    gt_box_xywh=gt_box,
                    status=STATUS_IDSW if is_idsw else STATUS_MATCH,
                    pred_id=pred_id,
                    pred_box_xywh=pred_box,
                    iou=float(similarity[row_idx, col_idx]),
                )
                prev_tracker_id[gt_id] = pred_id
                new_prev_timestep[gt_id] = pred_id
                matched_gt_ids.add(gt_id)

            for gt_id, gt_box in zip(gt_ids.tolist(), gt_boxes.tolist()):
                if gt_id in matched_gt_ids:
                    continue
                matches_t[gt_id] = TargetMatch(
                    gt_id=int(gt_id),
                    gt_box_xywh=tuple(float(v) for v in gt_box),
                    status=STATUS_LOST,
                    pred_id=None,
                    pred_box_xywh=None,
                    iou=0.0,
                )

            frame_matches[frame_id] = matches_t
            prev_timestep_tracker_id = new_prev_timestep
        return frame_matches

    def _select_render_targets(
        self,
        video_name: str,
        frame_infos: list[tuple[int, Path]],
        matches_by_tracker: list[dict[int, dict[int, TargetMatch]]],
    ) -> tuple[dict[int, set[int]], list[KeyEvent]]:
        frame_ids = [frame_id for frame_id, _ in frame_infos]
        if self.search_stable_clip:
            return {}, []
        if self.focus_gt_ids:
            focus_ids = set(self.focus_gt_ids)
            render_targets = {frame_id: set(focus_ids) for frame_id in frame_ids}
            return render_targets, []
        if not self.compare:
            render_targets = {
                frame_id: set(matches_by_tracker[0].get(frame_id, {}).keys())
                for frame_id in frame_ids
            }
            return render_targets, []

        primary_idx = self.primary_tracker_index if self.primary_tracker_index >= 0 else len(self.tracker_labels) + self.primary_tracker_index
        primary_idx = max(0, min(primary_idx, len(self.tracker_labels) - 1))
        raw_targets: dict[int, set[int]] = defaultdict(set)
        key_events: list[KeyEvent] = []
        for frame_id in frame_ids:
            primary_matches = matches_by_tracker[primary_idx].get(frame_id, {})
            if not primary_matches:
                continue
            for gt_id, primary_record in primary_matches.items():
                if self.primary_require_match:
                    if primary_record.status != STATUS_MATCH:
                        continue
                elif primary_record.status == STATUS_IDSW:
                    continue

                trigger_methods: list[str] = []
                for idx, method_matches in enumerate(matches_by_tracker):
                    if idx == primary_idx:
                        continue
                    status = method_matches.get(frame_id, {}).get(gt_id, None)
                    if status is None:
                        continue
                    if status.status in self.compare_statuses:
                        trigger_methods.append(self.tracker_labels[idx])
                if not trigger_methods:
                    continue
                raw_targets[frame_id].add(gt_id)
                key_events.append(
                    KeyEvent(
                        video_name=video_name,
                        frame_id=frame_id,
                        gt_id=gt_id,
                        primary_status=primary_record.status,
                        trigger_methods=trigger_methods,
                    )
                )

        expanded_targets: dict[int, set[int]] = defaultdict(set)
        if not raw_targets:
            return expanded_targets, key_events
        frame_to_index = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
        for key_frame_id, gt_ids in raw_targets.items():
            center_idx = frame_to_index[key_frame_id]
            begin = max(0, center_idx - self.context_frames)
            end = min(len(frame_ids) - 1, center_idx + self.context_frames)
            for idx in range(begin, end + 1):
                expanded_targets[frame_ids[idx]].update(gt_ids)
        return expanded_targets, key_events

    def _render_video(
        self,
        video_name: str,
        frame_infos: list[tuple[int, Path]],
        matches_by_tracker: list[dict[int, dict[int, TargetMatch]]],
        tracker_results_by_method: list[dict[int, np.ndarray]],
        render_targets_by_frame: dict[int, set[int]],
        progress=None,
    ) -> int:
        rendered = 0
        video_dir = self.output_dir / video_name
        video_dir.mkdir(parents=True, exist_ok=True)
        for frame_id, img_path in frame_infos:
            if self.max_frames > 0 and rendered >= self.max_frames:
                break
            target_frames = self._target_frame_filter(video_name)
            if target_frames is not None and frame_id not in target_frames:
                continue
            if self.search_stable_clip:
                if frame_id not in render_targets_by_frame:
                    continue
            elif not self.focus_gt_ids and self.compare and frame_id not in render_targets_by_frame:
                continue
            frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if frame is None:
                self._warn(f"video={video_name} frame={frame_id} image read failed: {img_path}")
                continue
            targets = render_targets_by_frame.get(frame_id, set())
            if self.search_stable_clip:
                visible_ids = set(targets)
            elif self.focus_gt_ids:
                visible_ids = set(self.focus_gt_ids)
            elif self.compare:
                visible_ids = set(targets)
                if self.render_non_key_targets:
                    visible_ids.update(matches_by_tracker[0].get(frame_id, {}).keys())
            else:
                visible_ids = set(matches_by_tracker[0].get(frame_id, {}).keys())

            row_gt_ids = self._select_table_rows(frame_id=frame_id, visible_ids=visible_ids, key_ids=targets, matches_by_tracker=matches_by_tracker)
            self._draw_frame(
                img=frame,
                video_name=video_name,
                frame_id=frame_id,
                visible_ids=visible_ids,
                table_gt_ids=row_gt_ids,
                total_visible_count=len(visible_ids),
                matches_by_tracker=matches_by_tracker,
                tracker_results_by_method=tracker_results_by_method,
                key_ids=targets,
            )
            out_path = video_dir / img_path.name
            cv2.imwrite(str(out_path), frame)
            rendered += 1
            if progress is not None:
                progress.set_postfix_str(f"{video_name}={frame_id:04d}", refresh=False)
                progress.update(1)
            elif rendered % 200 == 0:
                print(f"Offline compare render: {video_name}={frame_id:04d}")
        return rendered

    def _select_table_rows(
        self,
        frame_id: int,
        visible_ids: set[int],
        key_ids: set[int],
        matches_by_tracker: list[dict[int, dict[int, TargetMatch]]],
    ) -> list[int]:
        if self.search_stable_clip:
            gt_ids = sorted(int(gt_id) for gt_id in visible_ids)
            return gt_ids[:self.max_table_rows]
        if self.focus_gt_ids:
            return list(self.focus_gt_ids)[:self.max_table_rows]
        gt_ids = sorted(int(gt_id) for gt_id in visible_ids)
        if len(gt_ids) <= self.max_table_rows:
            return gt_ids

        def _sort_key(gt_id: int) -> tuple[int, float, int]:
            is_key = 0 if gt_id in key_ids else 1
            record = matches_by_tracker[0].get(frame_id, {}).get(gt_id, None)
            area = 0.0
            if record is not None:
                area = float(record.gt_box_xywh[2] * record.gt_box_xywh[3])
            return (is_key, -area, gt_id)

        return sorted(gt_ids, key=_sort_key)[:self.max_table_rows]

    def _draw_frame(
        self,
        img: np.ndarray,
        video_name: str,
        frame_id: int,
        visible_ids: set[int],
        table_gt_ids: list[int],
        total_visible_count: int,
        matches_by_tracker: list[dict[int, dict[int, TargetMatch]]],
        tracker_results_by_method: list[dict[int, np.ndarray]],
        key_ids: set[int],
    ):
        target_records = matches_by_tracker[0].get(frame_id, {})
        occupied_label_boxes: list[tuple[int, int, int, int]] = []
        for gt_id in sorted(visible_ids):
            gt_record = target_records.get(gt_id, None)
            if gt_record is None:
                continue
            self._draw_labeled_box(
                img=img,
                box_xywh=gt_record.gt_box_xywh,
                color=GT_COLOR_BGR,
                label="",
                thickness=self.line_thickness + (1 if gt_id in key_ids else 0),
            )
            for method_idx, method_matches in enumerate(matches_by_tracker):
                match = method_matches.get(frame_id, {}).get(gt_id, None)
                if match is None or match.pred_box_xywh is None:
                    continue
                self._draw_labeled_box(
                    img=img,
                    box_xywh=match.pred_box_xywh,
                    color=METHOD_COLORS_BGR[method_idx % len(METHOD_COLORS_BGR)],
                    label="",
                    thickness=self.line_thickness + (1 if gt_id in key_ids else 0),
                )
            anchor_rect = self._draw_anchor_id_stack(
                img=img,
                gt_record=gt_record,
                gt_id=gt_id,
                frame_id=frame_id,
                matches_by_tracker=matches_by_tracker,
            )
            if anchor_rect is not None:
                occupied_label_boxes.append(anchor_rect)
        if self.search_stable_clip and self.show_related_idsw_boxes and self.related_box_mode == "idsw":
            self._draw_related_idsw_boxes(
                img=img,
                video_name=video_name,
                frame_id=frame_id,
                matches_by_tracker=matches_by_tracker,
                tracker_results_by_method=tracker_results_by_method,
                occupied_label_boxes=occupied_label_boxes,
            )
        elif not self.focus_gt_ids and self.compare:
            self._draw_unmatched_predictions(
                img=img,
                frame_id=frame_id,
                matches_by_tracker=matches_by_tracker,
                tracker_results_by_method=tracker_results_by_method,
            )
        self._draw_header(
            img=img,
            video_name=video_name,
            frame_id=frame_id,
            table_gt_ids=table_gt_ids,
            total_visible_count=total_visible_count,
            matches_by_tracker=matches_by_tracker,
        )

    def _draw_anchor_id_stack(
        self,
        img: np.ndarray,
        gt_record: TargetMatch,
        gt_id: int,
        frame_id: int,
        matches_by_tracker: list[dict[int, dict[int, TargetMatch]]],
    ) -> tuple[int, int, int, int] | None:
        anchor_box = self._xywh_to_xyxy(box_xywh=gt_record.gt_box_xywh, width=img.shape[1], height=img.shape[0])
        if anchor_box is None:
            return None
        x1, y1, _, _ = anchor_box
        line_origin_x = x1 + 2
        line_origin_y = max(18, y1 - 6)
        segments: list[tuple[str, tuple[int, int, int]]] = [(str(gt_id), GT_COLOR_BGR)]
        for method_idx, method_matches in enumerate(matches_by_tracker):
            match = method_matches.get(frame_id, {}).get(gt_id, None)
            if match is None or match.pred_id is None:
                text = "-"
            else:
                text = str(match.pred_id)
            segments.append((text, METHOD_COLORS_BGR[method_idx % len(METHOD_COLORS_BGR)]))

        self._put_colored_segments(
            img=img,
            origin=(line_origin_x, line_origin_y),
            segments=segments,
            separator=" / ",
            separator_color=(232, 232, 232),
            scale=self.label_font_scale,
            thickness=1,
        )
        return self._colored_segments_rect(
            origin=(line_origin_x, line_origin_y),
            segments=segments,
            separator=" / ",
            scale=self.label_font_scale,
            thickness=1,
        )

    def _draw_related_idsw_boxes(
        self,
        img: np.ndarray,
        video_name: str,
        frame_id: int,
        matches_by_tracker: list[dict[int, dict[int, TargetMatch]]],
        tracker_results_by_method: list[dict[int, np.ndarray]],
        occupied_label_boxes: list[tuple[int, int, int, int]],
    ):
        related_pred_ids_by_method: dict[int, set[int]] = defaultdict(set)
        for candidate in self.selected_clip_candidates:
            if candidate.video_name != video_name:
                continue
            if frame_id < candidate.start_frame or frame_id > candidate.end_frame:
                continue
            for method_idx, pred_ids in candidate.related_pred_ids_by_method.items():
                related_pred_ids_by_method[method_idx].update(pred_ids)
        if not related_pred_ids_by_method:
            return

        for method_idx, pred_ids in related_pred_ids_by_method.items():
            tracker_rows = tracker_results_by_method[method_idx].get(frame_id, np.empty((0, 9), dtype=np.float32))
            if tracker_rows.size == 0:
                continue
            matched_pred_ids = {
                int(match.pred_id)
                for match in matches_by_tracker[method_idx].get(frame_id, {}).values()
                if match.pred_id is not None
            }
            color = METHOD_COLORS_BGR[method_idx % len(METHOD_COLORS_BGR)]
            for row in tracker_rows:
                pred_id = int(row[1])
                if pred_id not in pred_ids or pred_id in matched_pred_ids:
                    continue
                box_xywh = tuple(float(v) for v in row[2:6].tolist())
                label_origin = self._pick_non_overlapping_label_origin(
                    img=img,
                    box_xywh=box_xywh,
                    text=str(pred_id),
                    scale=self.label_font_scale,
                    thickness=1,
                    occupied_label_boxes=occupied_label_boxes,
                )
                self._draw_labeled_box(
                    img=img,
                    box_xywh=box_xywh,
                    color=color,
                    label=str(pred_id),
                    thickness=self.line_thickness,
                    label_origin=label_origin,
                )
                label_rect = self._text_rect(
                    text=str(pred_id),
                    origin=label_origin,
                    scale=self.label_font_scale,
                    thickness=1,
                )
                occupied_label_boxes.append(label_rect)

    def _draw_unmatched_predictions(
        self,
        img: np.ndarray,
        frame_id: int,
        matches_by_tracker: list[dict[int, dict[int, TargetMatch]]],
        tracker_results_by_method: list[dict[int, np.ndarray]],
    ):
        for method_idx, tracker_results in enumerate(tracker_results_by_method):
            tracker_rows = tracker_results.get(frame_id, np.empty((0, 9), dtype=np.float32))
            if tracker_rows.size == 0:
                continue
            matched_pred_ids = {
                int(match.pred_id)
                for match in matches_by_tracker[method_idx].get(frame_id, {}).values()
                if match.pred_id is not None
            }
            color = METHOD_COLORS_BGR[method_idx % len(METHOD_COLORS_BGR)]
            for row in tracker_rows:
                pred_id = int(row[1])
                if pred_id in matched_pred_ids:
                    continue
                self._draw_labeled_box(
                    img=img,
                    box_xywh=tuple(float(v) for v in row[2:6].tolist()),
                    color=color,
                    label=str(pred_id),
                    thickness=self.line_thickness,
                )

    def _draw_header(
        self,
        img: np.ndarray,
        video_name: str,
        frame_id: int,
        table_gt_ids: list[int],
        total_visible_count: int,
        matches_by_tracker: list[dict[int, dict[int, TargetMatch]]],
    ):
        row_h = 24
        first_col_w = 86
        col_w = 74
        table_rows = 2 + len(table_gt_ids)
        table_h = 14 + row_h * table_rows
        table_w = first_col_w + col_w * len(self.tracker_labels)
        x0, y0 = 10, 10
        self._blend_region(img, x0, y0, x0 + table_w, y0 + table_h, (20, 20, 20), 0.58)
        self._put_text(img, f"{video_name} / {frame_id:04d}", (x0 + 10, y0 + 18), (245, 245, 245), 0.58, 1)

        y_header = y0 + 14 + row_h
        self._put_text(img, "GT", (x0 + 10, y_header), GT_COLOR_BGR, 0.54, 1)
        for idx, label in enumerate(self.tracker_labels):
            text_x = x0 + first_col_w + idx * col_w + 6
            self._put_text(img, label[:8], (text_x, y_header), METHOD_COLORS_BGR[idx % len(METHOD_COLORS_BGR)], 0.50, 1)

        for row_idx, gt_id in enumerate(table_gt_ids, start=2):
            text_y = y0 + 14 + row_h * row_idx
            self._put_text(img, f"GT {gt_id}", (x0 + 10, text_y), (240, 240, 240), 0.50, 1)
            gt_present = gt_id in matches_by_tracker[0].get(frame_id, {})
            for method_idx, method_matches in enumerate(matches_by_tracker):
                match = method_matches.get(frame_id, {}).get(gt_id, None)
                if not gt_present:
                    status = STATUS_ABSENT
                else:
                    status = STATUS_LOST if match is None else match.status
                cell_x = x0 + first_col_w + method_idx * col_w + 20
                self._put_text(img, STATUS_LABELS[status], (cell_x, text_y), STATUS_COLORS[status], 0.58, 2)
        hidden = max(0, int(total_visible_count) - len(table_gt_ids))
        if hidden > 0:
            self._put_text(img, f"...+{hidden}", (x0 + 10, y0 + table_h - 4), (210, 210, 210), 0.46, 1)

    @staticmethod
    def _method_tag(method_idx: int) -> str:
        return ["A", "B", "C", "D", "E"][method_idx] if method_idx < 5 else f"M{method_idx + 1}"

    def _draw_labeled_box(
        self,
        img: np.ndarray,
        box_xywh: tuple[float, float, float, float],
        color: tuple[int, int, int],
        label: str,
        thickness: int,
        label_origin: tuple[int, int] | None = None,
    ):
        box = self._xywh_to_xyxy(box_xywh=box_xywh, width=img.shape[1], height=img.shape[0])
        if box is None:
            return
        overlay = img.copy()
        x1, y1, x2, y2 = box
        # Add a dark under-stroke so the colored bbox keeps its hue but separates from busy stage backgrounds.
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 12, 12), thickness + 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.addWeighted(overlay, self.border_alpha, img, 1.0 - self.border_alpha, 0.0, dst=img)
        if label:
            origin = label_origin if label_origin is not None else (x1, max(14, y1 - 4))
            self._put_text(img, label, origin, color, self.label_font_scale, 1)

    @staticmethod
    def _xywh_to_xyxy(box_xywh: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
        x, y, w, h = box_xywh
        x1 = max(0, min(int(np.floor(x)), width - 1))
        y1 = max(0, min(int(np.floor(y)), height - 1))
        x2 = max(0, min(int(np.ceil(x + w)), width - 1))
        y2 = max(0, min(int(np.ceil(y + h)), height - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _blend_region(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], alpha: float):
        if alpha <= 0.0 or x2 <= x1 or y2 <= y1:
            return
        x1 = max(0, min(x1, img.shape[1]))
        x2 = max(0, min(x2, img.shape[1]))
        y1 = max(0, min(y1, img.shape[0]))
        y2 = max(0, min(y2, img.shape[0]))
        if x2 <= x1 or y2 <= y1:
            return
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return
        color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        roi[...] = np.clip(roi.astype(np.float32) * (1.0 - alpha) + color_arr * alpha, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _put_text(
        img: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        scale: float,
        thickness: int,
    ):
        cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (16, 16, 16), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _put_colored_segments(
        self,
        img: np.ndarray,
        origin: tuple[int, int],
        segments: list[tuple[str, tuple[int, int, int]]],
        separator: str,
        separator_color: tuple[int, int, int],
        scale: float,
        thickness: int,
    ):
        cursor_x, cursor_y = origin
        for seg_idx, (text, color) in enumerate(segments):
            self._put_text(img, text, (cursor_x, cursor_y), color, scale, thickness)
            cursor_x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
            if seg_idx == len(segments) - 1:
                continue
            self._put_text(img, separator, (cursor_x, cursor_y), separator_color, scale, thickness)
            cursor_x += cv2.getTextSize(separator, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]

    def _colored_segments_rect(
        self,
        origin: tuple[int, int],
        segments: list[tuple[str, tuple[int, int, int]]],
        separator: str,
        scale: float,
        thickness: int,
    ) -> tuple[int, int, int, int]:
        full_text = separator.join(text for text, _ in segments)
        return self._text_rect(
            text=full_text,
            origin=origin,
            scale=scale,
            thickness=thickness,
        )

    @staticmethod
    def _text_rect(
        text: str,
        origin: tuple[int, int],
        scale: float,
        thickness: int,
    ) -> tuple[int, int, int, int]:
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = origin
        return x, y - text_h - baseline, x + text_w, y + baseline

    def _pick_non_overlapping_label_origin(
        self,
        img: np.ndarray,
        box_xywh: tuple[float, float, float, float],
        text: str,
        scale: float,
        thickness: int,
        occupied_label_boxes: list[tuple[int, int, int, int]],
    ) -> tuple[int, int]:
        box = self._xywh_to_xyxy(box_xywh=box_xywh, width=img.shape[1], height=img.shape[0])
        if box is None:
            return 0, 14
        x1, y1, x2, y2 = box
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        right_x = max(0, min(x2 - text_w, img.shape[1] - text_w - 1))
        bottom_y = min(img.shape[0] - 4, y2 + 14)
        candidates = [
            (x1, max(14, y1 - 4)),
            (right_x, max(14, y1 - 4)),
            (x1, bottom_y),
            (right_x, bottom_y),
        ]
        for origin in candidates:
            rect = self._text_rect(text=text, origin=origin, scale=scale, thickness=thickness)
            if not any(self._rect_overlap(rect, occupied_rect) for occupied_rect in occupied_label_boxes):
                return origin
        return candidates[0]

    @staticmethod
    def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

    def _build_summary_rows(
        self,
        video_name: str,
        matches_by_tracker: list[dict[int, dict[int, TargetMatch]]],
    ) -> list[dict[str, str | int | float]]:
        rows: list[dict[str, str | int | float]] = []
        for method_idx, method_label in enumerate(self.tracker_labels):
            method_matches = matches_by_tracker[method_idx]
            for frame_id, targets in method_matches.items():
                for gt_id, match in targets.items():
                    rows.append(
                        {
                            "video": video_name,
                            "frame": frame_id,
                            "gt_id": gt_id,
                            "method": method_label,
                            "status": match.status,
                            "pred_id": "" if match.pred_id is None else match.pred_id,
                            "iou": round(match.iou, 4),
                        }
                    )
        return rows

    def _search_stable_clips(self, video_plans: list[VideoRenderPlan]) -> list[StableClipCandidate]:
        if len(self.tracker_labels) < 3:
            return []
        primary_idx = self.primary_tracker_index if self.primary_tracker_index >= 0 else len(self.tracker_labels) + self.primary_tracker_index
        primary_idx = max(0, min(primary_idx, len(self.tracker_labels) - 1))
        secondary_indices = [idx for idx in range(len(self.tracker_labels)) if idx != primary_idx]
        candidates: list[StableClipCandidate] = []

        for video_plan in video_plans:
            frame_ids = [frame_id for frame_id, _ in video_plan.frame_infos]
            if len(frame_ids) < self.search_window:
                continue
            all_gt_ids = sorted({gt_id for frame_matches in video_plan.matches_by_tracker[primary_idx].values() for gt_id in frame_matches.keys()})
            for gt_id in all_gt_ids:
                start_idx = 0
                while start_idx + self.search_window <= len(frame_ids):
                    window_frame_ids = frame_ids[start_idx:start_idx + self.search_window]
                    primary_window = [video_plan.matches_by_tracker[primary_idx].get(frame_id, {}).get(gt_id, None) for frame_id in window_frame_ids]
                    if any(match is None or match.status != STATUS_MATCH or match.pred_id is None for match in primary_window):
                        start_idx += 1
                        continue
                    primary_pred_ids = {int(match.pred_id) for match in primary_window if match is not None and match.pred_id is not None}
                    if len(primary_pred_ids) != 1:
                        start_idx += 1
                        continue

                    related_pred_ids_by_method: dict[int, tuple[int, ...]] = {}
                    switched_methods: list[str] = []
                    window_key_frames: set[int] = set()
                    for method_idx in secondary_indices:
                        method_window = [video_plan.matches_by_tracker[method_idx].get(frame_id, {}).get(gt_id, None) for frame_id in window_frame_ids]
                        pred_ids = [int(match.pred_id) for match in method_window if match is not None and match.pred_id is not None]
                        unique_pred_ids = tuple(sorted(set(pred_ids)))
                        method_key_frames = tuple(
                            frame_id
                            for frame_id, match in zip(window_frame_ids, method_window)
                            if match is not None and match.status == STATUS_IDSW
                        )
                        has_idsw = len(method_key_frames) > 0
                        if len(unique_pred_ids) < 2 or not has_idsw:
                            switched_methods = []
                            break
                        related_pred_ids_by_method[method_idx] = unique_pred_ids
                        switched_methods.append(self.tracker_labels[method_idx])
                        window_key_frames.update(method_key_frames)

                    if not switched_methods or not window_key_frames:
                        start_idx += 1
                        continue

                    clip_start_frame, clip_end_frame = self._resolve_clip_bounds(
                        frame_ids=frame_ids,
                        key_frames=sorted(window_key_frames),
                        clip_len=self.search_window,
                    )

                    candidates.append(
                        StableClipCandidate(
                            video_name=video_plan.video_name,
                            gt_id=gt_id,
                            start_frame=clip_start_frame,
                            end_frame=clip_end_frame,
                            primary_pred_id=next(iter(primary_pred_ids)),
                            key_frames=tuple(sorted(window_key_frames)),
                            related_pred_ids_by_method=related_pred_ids_by_method,
                            switched_methods=tuple(switched_methods),
                        )
                    )
                    start_idx += self.search_window

        candidates.sort(key=lambda item: (item.video_name, item.start_frame, item.gt_id))
        if self.search_limit < 0:
            return candidates
        return candidates[:self.search_limit]

    @staticmethod
    def _resolve_clip_bounds(frame_ids: list[int], key_frames: list[int], clip_len: int) -> tuple[int, int]:
        if not frame_ids:
            raise ValueError("frame_ids must not be empty")
        clip_len = max(1, clip_len)
        if clip_len >= len(frame_ids):
            return frame_ids[0], frame_ids[-1]

        frame_to_index = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
        key_indices = [frame_to_index[frame_id] for frame_id in key_frames if frame_id in frame_to_index]
        if not key_indices:
            center_idx = clip_len // 2
        else:
            event_start_idx = min(key_indices)
            event_end_idx = max(key_indices)
            center_idx = (event_start_idx + event_end_idx) // 2

        start_idx = max(0, center_idx - clip_len // 2)
        end_idx = start_idx + clip_len - 1
        if end_idx >= len(frame_ids):
            end_idx = len(frame_ids) - 1
            start_idx = max(0, end_idx - clip_len + 1)
        return frame_ids[start_idx], frame_ids[end_idx]

    def _apply_search_candidates(self, video_plans: list[VideoRenderPlan], candidates: list[StableClipCandidate]):
        targets_by_video: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
        for candidate in candidates:
            for frame_id in range(candidate.start_frame, candidate.end_frame + 1):
                targets_by_video[candidate.video_name][frame_id].add(candidate.gt_id)
        for video_plan in video_plans:
            video_targets = targets_by_video.get(video_plan.video_name, {})
            video_plan.render_targets_by_frame = {frame_id: set(gt_ids) for frame_id, gt_ids in video_targets.items()}
            video_plan.planned_frames = self._count_planned_frames(
                video_name=video_plan.video_name,
                frame_infos=video_plan.frame_infos,
                render_targets_by_frame=video_plan.render_targets_by_frame,
            )

    def _write_summary(
        self,
        selected_videos: list[str],
        rendered_videos: int,
        rendered_frames: int,
        key_events: list[KeyEvent],
        summary_rows: list[dict[str, str | int | float]],
    ):
        run_summary_lines = [
            f"compare: {self.compare}",
            f"focus_gt_ids: {list(self.focus_gt_ids) if self.focus_gt_ids else None}",
            f"target_frames: {self._format_target_frames() if self.target_frames_by_video else None}",
            f"search_stable_clip: {self.search_stable_clip}",
            f"search_window: {self.search_window}",
            f"show_related_idsw_boxes: {self.show_related_idsw_boxes}",
            f"related_box_mode: {self.related_box_mode}",
            f"iou_threshold: {self.iou_threshold}",
            f"context_frames: {self.context_frames}",
            f"output_dir: {self.output_dir}",
            f"selected_videos: {', '.join(selected_videos)}",
            f"rendered_videos: {rendered_videos}",
            f"rendered_frames: {rendered_frames}",
            f"stable_clip_candidates: {len(self.selected_clip_candidates)}",
            f"key_events: {len(key_events)}",
        ]
        for idx, (label, tracker_dir) in enumerate(zip(self.tracker_labels, self.tracker_dirs), start=1):
            run_summary_lines.append(f"tracker_{idx}_label: {label}")
            run_summary_lines.append(f"tracker_{idx}_dir: {tracker_dir}")
        (self.output_dir / "run_summary.txt").write_text("\n".join(run_summary_lines) + "\n", encoding="utf-8")

        with open(self.output_dir / "compare_events.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "frame", "gt_id", "primary_status", "trigger_methods"])
            for event in key_events:
                writer.writerow([event.video_name, event.frame_id, event.gt_id, event.primary_status, "|".join(event.trigger_methods)])

        with open(self.output_dir / "match_status.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video", "frame", "gt_id", "method", "status", "pred_id", "iou"])
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        with open(self.output_dir / "stable_clip_candidates.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "gt_id", "start_frame", "end_frame", "primary_pred_id", "key_frames", "switched_methods", "related_pred_ids"])
            for candidate in self.selected_clip_candidates:
                related_pred_ids = ";".join(
                    f"{self.tracker_labels[idx]}:{'|'.join(str(pred_id) for pred_id in pred_ids)}"
                    for idx, pred_ids in sorted(candidate.related_pred_ids_by_method.items())
                )
                writer.writerow([
                    candidate.video_name,
                    candidate.gt_id,
                    candidate.start_frame,
                    candidate.end_frame,
                    candidate.primary_pred_id,
                    "|".join(str(frame_id) for frame_id in candidate.key_frames),
                    "|".join(candidate.switched_methods),
                    related_pred_ids,
                ])

    @staticmethod
    def _warn(msg: str):
        print(f"[Error警告] {msg}")
