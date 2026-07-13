from __future__ import annotations

import io
import json
import os
import queue
import threading
from dataclasses import dataclass
from typing import Optional
import zipfile

import cv2
import numpy as np
import torch

from ..core import BaseVisualTask, FrameContext, HookEvent


@dataclass
class _BundleFrameJob:
    video_name: str
    bundle_path: str
    frame_record: dict
    heat_map_u8: np.ndarray


class _AsyncBundleWriter:
    SUMMARY_FILENAME = "summary.json"

    def __init__(self, chunk_size: int = 64):
        self.chunk_size = max(1, int(chunk_size))
        self._queue: queue.SimpleQueue[Optional[_BundleFrameJob]] = queue.SimpleQueue()
        self._worker = threading.Thread(target=self._worker_loop, name="focus-bundle-writer", daemon=True)
        self._worker.start()

    def submit(self, job: _BundleFrameJob):
        self._queue.put(job)

    def close(self):
        self._queue.put(None)
        self._worker.join()

    def _worker_loop(self):
        states: dict[str, dict] = {}
        while True:
            job = self._queue.get()
            if job is None:
                break

            state = states.get(job.video_name, None)
            if state is None:
                writer = zipfile.ZipFile(job.bundle_path, mode="w", compression=zipfile.ZIP_STORED)
                state = {
                    "writer": writer,
                    "frames": [],
                    "chunk_idx": 0,
                    "pending_heat_maps": [],
                    "pending_records": [],
                }
                states[job.video_name] = state

            state["pending_records"].append(dict(job.frame_record))
            if isinstance(job.heat_map_u8, np.ndarray) and job.heat_map_u8.size > 0:
                state["pending_heat_maps"].append(np.asarray(job.heat_map_u8, dtype=np.uint8, order="C"))
            else:
                state["pending_heat_maps"].append(np.zeros((0, 0), dtype=np.uint8))

            if len(state["pending_records"]) >= self.chunk_size:
                self._flush_chunk(video_name=job.video_name, state=state)

        for video_name, state in states.items():
            self._flush_chunk(video_name=video_name, state=state)
            summary = {
                "video_name": str(video_name),
                "task_name": "decoder_l0_query_focus",
                "num_frames": int(len(state["frames"])),
                "chunk_size": int(self.chunk_size),
                "frames": state["frames"],
            }
            state["writer"].writestr(self.SUMMARY_FILENAME, json.dumps(summary, ensure_ascii=False))
            state["writer"].close()

    def _flush_chunk(self, video_name: str, state: dict):
        pending_records = state["pending_records"]
        pending_heat_maps = state["pending_heat_maps"]
        if not pending_records:
            return

        chunk_rel_path = f"heat_chunks/chunk_{int(state['chunk_idx']):05d}.npz"
        valid_positions = [
            idx for idx, heat in enumerate(pending_heat_maps) if isinstance(heat, np.ndarray) and heat.size > 0
        ]
        valid_heat_maps = [pending_heat_maps[idx] for idx in valid_positions]
        heat_index_lookup = {src_idx: heat_idx for heat_idx, src_idx in enumerate(valid_positions)}

        if valid_heat_maps:
            buf = io.BytesIO()
            max_stem_len = max(len(str(record.get("stem", ""))) for record in pending_records)
            np.savez_compressed(
                buf,
                heat_maps=np.stack(valid_heat_maps, axis=0).astype(np.uint8, copy=False),
                frame_ids=np.asarray([int(record["frame_id"]) for record in pending_records], dtype=np.int32),
                stems=np.asarray(
                    [str(record.get("stem", "")) for record in pending_records],
                    dtype=f"<U{max(1, max_stem_len)}",
                ),
            )
            state["writer"].writestr(chunk_rel_path, buf.getvalue(), compress_type=zipfile.ZIP_STORED)

        for idx, (record, heat_map) in enumerate(zip(pending_records, pending_heat_maps)):
            if idx in heat_index_lookup:
                record["heat_chunk_path"] = chunk_rel_path
                record["heat_chunk_index"] = int(heat_index_lookup[idx])
                record["heat_shape"] = [int(v) for v in heat_map.shape[:2]]
            else:
                record["heat_chunk_path"] = ""
                record["heat_chunk_index"] = -1
                record["heat_shape"] = [0, 0]
            state["frames"].append(record)

        state["pending_records"] = []
        state["pending_heat_maps"] = []
        state["chunk_idx"] += 1


class DecoderL0QueryFocusTask(BaseVisualTask):
    HOOK_FOCUS = "decoder_l0_det_query_focus"
    BUNDLE_FILENAME = "video_bundle.zip"
    SUMMARY_FILENAME = "summary.json"

    def __init__(self, task_name: str, cfg: dict, mode: str, root_dir: str):
        super().__init__(task_name, cfg, mode, root_dir)
        self.save_image = bool(cfg.get("save_image", True))
        self.show_image = bool(cfg.get("show_image", False))
        self.window_delay = int(cfg.get("window_delay", 1))
        self.alpha = float(np.clip(float(cfg.get("alpha", 0.42)), 0.0, 1.0))
        self.min_score = float(cfg.get("min_score", 0.0))
        self.topk_queries = max(1, int(cfg.get("topk_queries", 12)))
        self.draw_frame_text = bool(cfg.get("draw_frame_text", False))
        self.draw_prev_track_bbox = bool(cfg.get("draw_prev_track_bbox", True))
        self.draw_missing_gt_bbox = bool(cfg.get("draw_missing_gt_bbox", True))
        self.prev_track_bbox_alpha = float(np.clip(float(cfg.get("prev_track_bbox_alpha", 0.60)), 0.0, 1.0))
        self.prev_track_bbox_thickness = max(1, int(cfg.get("prev_track_bbox_thickness", 3)))
        self.missing_gt_bbox_thickness = max(1, int(cfg.get("missing_gt_bbox_thickness", 3)))
        self.missing_gt_iou_threshold = float(np.clip(float(cfg.get("missing_gt_iou_threshold", 0.5)), 0.0, 1.0))
        self.gt_visibility_threshold = float(cfg.get("gt_visibility_threshold", 0.0))
        self.aggregate_mode = str(cfg.get("aggregate_mode", "sum")).strip().lower()
        self.blur_kernel = max(0, int(cfg.get("blur_kernel", 21)))
        self.blur_sigma = float(max(0.0, cfg.get("blur_sigma", 0.0)))
        self.norm_percentile = float(np.clip(float(cfg.get("norm_percentile", 99.0)), 50.0, 100.0))
        self.level_weight_mode = str(cfg.get("level_weight_mode", "uniform")).strip().lower()
        self.bundle_chunk_size = max(1, int(cfg.get("bundle_chunk_size", 64)))
        self._last_track_result_by_video: dict[str, np.ndarray] = {}
        self._gt_boxes_by_video: dict[str, dict[int, np.ndarray]] = {}
        self._bundle_writer = _AsyncBundleWriter(chunk_size=self.bundle_chunk_size)
        self._state_lock = threading.Lock()

    def required_switches(self) -> set[str]:
        return {self.HOOK_FOCUS}

    def requires_image(self) -> bool:
        return bool(self.enabled and (self.save_image or self.show_image))

    def update(self, frame: FrameContext, hook_events: list[HookEvent]):
        if not self.enabled:
            return

        focus_event = self._latest_event(hook_events, self.HOOK_FOCUS)
        if focus_event is None:
            return

        img = frame.get_image_bgr()
        if img is None:
            return

        focus_data = self._extract_focus_data(focus_event.payload, image_shape=img.shape[:2])
        if focus_data is None:
            return

        with self._state_lock:
            prev_track_result = self._last_track_result_by_video.get(
                frame.video_name, np.empty((0, 9), dtype=np.float32)
            ).copy()
            all_gt_boxes = self._load_gt_boxes_for_video(frame).get(int(frame.frame_id), np.zeros((0, 4), dtype=np.float32))

        focus_gt_boxes = self._find_focus_gt_boxes(frame=frame, prev_track_result=prev_track_result, gt_boxes=all_gt_boxes)
        out_img = np.ascontiguousarray(img.copy())
        if focus_data["heat_overlay"].size > 0:
            out_img = self._blend_heat_overlay(out_img, focus_data["heat_overlay"])

        if self.draw_prev_track_bbox:
            self._draw_prev_track_bbox(out_img, prev_track_result)
        if self.draw_missing_gt_bbox:
            self._draw_missing_gt_bbox(out_img, focus_gt_boxes)

        if self.draw_frame_text:
            cv2.putText(
                out_img,
                (
                    f"Frame: {frame.frame_id}  L0 det queries: {focus_data['num_valid_queries']}  "
                    f"prev tracks: {int(prev_track_result.shape[0])}  focus gt: {int(focus_gt_boxes.shape[0])}"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
            )

        if self.save_image:
            self.submit_image(frame, out_img)
        with self._state_lock:
            self._last_track_result_by_video[frame.video_name] = np.asarray(frame.track_result, dtype=np.float32).copy()
        self._append_video_bundle(
            frame=frame,
            focus_data=focus_data,
            prev_track_result=prev_track_result,
            focus_gt_boxes=focus_gt_boxes,
            all_gt_boxes=all_gt_boxes,
        )

        if self.show_image:
            cv2.imshow(f"{frame.video_name}:{self.task_name}", out_img)
            cv2.waitKey(self.window_delay)

    def close(self):
        super().close()
        if self._bundle_writer is not None:
            self._bundle_writer.close()
        if self.show_image:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    @staticmethod
    def _latest_event(events: list[HookEvent], name: str) -> Optional[HookEvent]:
        for event in reversed(events):
            if event.name == name:
                return event
        return None

    def _find_focus_gt_boxes(
        self,
        frame: FrameContext,
        prev_track_result: np.ndarray,
        gt_boxes: np.ndarray | None = None,
    ) -> np.ndarray:
        if gt_boxes is None:
            gt_by_frame = self._load_gt_boxes_for_video(frame)
            gt_boxes = gt_by_frame.get(int(frame.frame_id), np.zeros((0, 4), dtype=np.float32))
        if gt_boxes.size == 0:
            return gt_boxes

        track_boxes = self._track_boxes_to_xyxy(prev_track_result)
        if track_boxes.size == 0:
            # When the previous frame has no active tracks, every current GT box is a target
            # that detection queries should consider, including the first frame of the video.
            return gt_boxes

        iou = self._box_iou_matrix(gt_boxes, track_boxes)
        keep = iou.max(axis=1) < self.missing_gt_iou_threshold
        return gt_boxes[keep]

    def _load_gt_boxes_for_video(self, frame: FrameContext) -> dict[int, np.ndarray]:
        cached = self._gt_boxes_by_video.get(frame.video_name, None)
        if cached is not None:
            return cached

        video_dir = os.path.dirname(os.path.dirname(frame.img_path))
        gt_path = os.path.join(video_dir, "gt", "gt.txt")
        gt_by_frame: dict[int, list[list[float]]] = {}
        if not os.path.isfile(gt_path):
            self._gt_boxes_by_video[frame.video_name] = {}
            return self._gt_boxes_by_video[frame.video_name]

        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split(",")
                if len(fields) < 6:
                    continue
                try:
                    frame_id = int(float(fields[0]))
                    x = float(fields[2])
                    y = float(fields[3])
                    w = float(fields[4])
                    h = float(fields[5])
                    considered = float(fields[6]) if len(fields) > 6 else 1.0
                    cls = int(float(fields[7])) if len(fields) > 7 else 1
                    vis = float(fields[8]) if len(fields) > 8 else 1.0
                except ValueError:
                    continue

                if considered <= 0.0 or cls <= 0 or vis < self.gt_visibility_threshold or w <= 0.0 or h <= 0.0:
                    continue
                gt_by_frame.setdefault(frame_id, []).append([x, y, x + w, y + h])

        self._gt_boxes_by_video[frame.video_name] = {
            frame_id: np.asarray(boxes, dtype=np.float32) for frame_id, boxes in gt_by_frame.items()
        }
        return self._gt_boxes_by_video[frame.video_name]

    def _extract_focus_data(self, payload: dict, image_shape: tuple[int, int]) -> Optional[dict[str, np.ndarray | int]]:
        sampling_locations = payload.get("sampling_locations", None)
        attention_weights = payload.get("attention_weights", None)
        query_logits = payload.get("query_logits", None)
        query_boxes_after = payload.get("query_boxes_after", None)
        query_mask = payload.get("query_mask", None)
        value_spatial_shapes = payload.get("value_spatial_shapes", None)
        if not all(
            isinstance(x, torch.Tensor)
            for x in (sampling_locations, attention_weights, query_logits, query_boxes_after, value_spatial_shapes)
        ):
            return None

        loc = sampling_locations.detach().to(device="cpu", dtype=torch.float32)
        attn = attention_weights.detach().to(device="cpu", dtype=torch.float32)
        logits = query_logits.detach().to(device="cpu", dtype=torch.float32)
        boxes = query_boxes_after.detach().to(device="cpu", dtype=torch.float32)
        spatial_shapes = value_spatial_shapes.detach().to(device="cpu", dtype=torch.long)

        if loc.ndim != 6 or attn.ndim != 5 or logits.ndim != 3 or boxes.ndim != 3 or spatial_shapes.ndim != 2:
            return None
        if loc.shape[0] == 0 or logits.shape[0] == 0 or boxes.shape[0] == 0:
            return None

        loc = loc[0]
        attn = attn[0]
        logits = logits[0]
        boxes = boxes[0]

        if isinstance(query_mask, torch.Tensor) and query_mask.ndim >= 2 and query_mask.shape[0] > 0:
            valid_mask = ~query_mask[0].detach().to(device="cpu", dtype=torch.bool)
        else:
            valid_mask = torch.ones((logits.shape[0],), dtype=torch.bool)

        scores = torch.sigmoid(logits).amax(dim=-1)
        if self.min_score > 0.0:
            valid_mask &= scores >= self.min_score

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return None

        valid_scores = scores.index_select(0, valid_indices)
        topk = min(self.topk_queries, int(valid_scores.numel()))
        topk_rel = torch.topk(valid_scores, k=topk, largest=True, sorted=True).indices
        topk_indices = valid_indices.index_select(0, topk_rel)

        topk_scores = scores.index_select(0, topk_indices).numpy().astype(np.float32)
        topk_boxes = boxes.index_select(0, topk_indices)
        topk_boxes_xyxy = self._cxcywh_to_xyxy(topk_boxes, image_shape=image_shape)
        topk_locs = loc.index_select(0, topk_indices).numpy().astype(np.float32)
        topk_attn = attn.index_select(0, topk_indices).numpy().astype(np.float32)
        spatial_shapes_np = spatial_shapes.numpy().astype(np.int32)

        level_maps, heat_map = self._build_focus_heat_map(
            image_shape=image_shape,
            spatial_shapes=spatial_shapes_np,
            sampling_locations=topk_locs,
            attention_weights=topk_attn,
            query_scores=topk_scores,
        )
        heat_overlay = self._colorize_heat_map(heat_map)

        return {
            "num_valid_queries": int(valid_indices.numel()),
            "topk_indices": topk_indices.numpy().astype(np.int32),
            "topk_scores": topk_scores,
            "topk_boxes_xyxy": topk_boxes_xyxy,
            "topk_boxes_norm": topk_boxes.numpy().astype(np.float32),
            "topk_sampling_locations": topk_locs,
            "topk_attention_weights": topk_attn,
            "value_spatial_shapes": spatial_shapes_np,
            "per_level_heat": level_maps,
            "heat_map": heat_map,
            "heat_overlay": heat_overlay,
        }

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor, image_shape: tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        boxes = boxes.clamp(0.0, 1.0)
        scale = boxes.new_tensor([w, h, w, h], dtype=torch.float32)
        boxes = boxes * scale
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        bw = boxes[:, 2]
        bh = boxes[:, 3]
        x1 = torch.clamp(cx - bw * 0.5, min=0.0, max=max(0.0, float(w - 1)))
        y1 = torch.clamp(cy - bh * 0.5, min=0.0, max=max(0.0, float(h - 1)))
        x2 = torch.clamp(cx + bw * 0.5, min=0.0, max=float(w - 1))
        y2 = torch.clamp(cy + bh * 0.5, min=0.0, max=float(h - 1))
        return torch.stack([x1, y1, x2, y2], dim=1).round().to(dtype=torch.int32).numpy()

    def _build_focus_heat_map(
        self,
        image_shape: tuple[int, int],
        spatial_shapes: np.ndarray,
        sampling_locations: np.ndarray,
        attention_weights: np.ndarray,
        query_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._build_splat_heat_map(
            image_shape=image_shape,
            spatial_shapes=spatial_shapes,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            query_scores=query_scores,
        )

    def _build_splat_heat_map(
        self,
        image_shape: tuple[int, int],
        spatial_shapes: np.ndarray,
        sampling_locations: np.ndarray,
        attention_weights: np.ndarray,
        query_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        level_canvases = [np.zeros((int(h), int(w)), dtype=np.float32) for h, w in spatial_shapes.tolist()]
        level_weights = self._resolve_level_weights(spatial_shapes)

        num_queries, num_heads, num_levels, num_points = attention_weights.shape
        for query_idx in range(num_queries):
            score = max(float(query_scores[query_idx]), 1e-6)
            for head_idx in range(num_heads):
                for level_idx in range(num_levels):
                    level_h = int(spatial_shapes[level_idx, 0])
                    level_w = int(spatial_shapes[level_idx, 1])
                    canvas = level_canvases[level_idx]
                    level_weight = level_weights[level_idx]
                    for point_idx in range(num_points):
                        x_norm = float(sampling_locations[query_idx, head_idx, level_idx, point_idx, 0])
                        y_norm = float(sampling_locations[query_idx, head_idx, level_idx, point_idx, 1])
                        weight = float(attention_weights[query_idx, head_idx, level_idx, point_idx]) * score * level_weight
                        self._splat_point(canvas, x_norm=x_norm, y_norm=y_norm, value=weight, width=level_w, height=level_h)

        return self._resize_and_aggregate_level_maps(level_canvases, image_shape=image_shape)

    @staticmethod
    def _splat_point(canvas: np.ndarray, x_norm: float, y_norm: float, value: float, width: int, height: int):
        if value <= 0.0 or width <= 0 or height <= 0:
            return

        x = float(np.clip(x_norm, 0.0, 1.0)) * width - 0.5
        y = float(np.clip(y_norm, 0.0, 1.0)) * height - 0.5

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1

        wx1 = x - x0
        wy1 = y - y0
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        for xx, wx in ((x0, wx0), (x1, wx1)):
            if xx < 0 or xx >= width or wx <= 0.0:
                continue
            for yy, wy in ((y0, wy0), (y1, wy1)):
                if yy < 0 or yy >= height or wy <= 0.0:
                    continue
                canvas[yy, xx] += float(value * wx * wy)

    def _resolve_level_weights(self, spatial_shapes: np.ndarray) -> np.ndarray:
        if self.level_weight_mode == "area_inv":
            areas = np.maximum(spatial_shapes[:, 0] * spatial_shapes[:, 1], 1)
            inv = 1.0 / areas.astype(np.float32)
            return inv / max(float(inv.sum()), 1e-6)
        return np.ones((spatial_shapes.shape[0],), dtype=np.float32)

    def _resize_and_aggregate_level_maps(
        self,
        level_maps: list[np.ndarray],
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_shape
        resized_maps = []
        for level_map in level_maps:
            if level_map.size == 0:
                resized = np.zeros((h, w), dtype=np.float32)
            else:
                resized = cv2.resize(level_map, (w, h), interpolation=cv2.INTER_LINEAR)
            resized_maps.append(resized.astype(np.float32, copy=False))

        stacked = np.stack(resized_maps, axis=0) if resized_maps else np.zeros((0, h, w), dtype=np.float32)
        if stacked.size == 0:
            return np.zeros((0, h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        if self.aggregate_mode == "max":
            heat = stacked.max(axis=0)
        else:
            heat = stacked.sum(axis=0)

        if self.blur_kernel > 1:
            kernel = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
            heat = cv2.GaussianBlur(heat, (kernel, kernel), sigmaX=self.blur_sigma, sigmaY=self.blur_sigma)
            stacked = np.stack(
                [
                    cv2.GaussianBlur(level_map, (kernel, kernel), sigmaX=self.blur_sigma, sigmaY=self.blur_sigma)
                    for level_map in stacked
                ],
                axis=0,
            )

        return stacked.astype(np.float32, copy=False), heat.astype(np.float32, copy=False)

    def _colorize_heat_map(self, heat_map: np.ndarray) -> np.ndarray:
        if heat_map.size == 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        max_val = float(heat_map.max())
        if max_val <= 0.0:
            return np.zeros((heat_map.shape[0], heat_map.shape[1], 3), dtype=np.uint8)

        clip_val = float(np.percentile(heat_map, self.norm_percentile))
        clip_val = max(clip_val, max_val * 1e-6)
        heat = np.clip(heat_map / clip_val, 0.0, 1.0)
        heat_u8 = np.round(heat * 255.0).astype(np.uint8)
        # Use viridis for query-focus heatmaps to improve perceptual uniformity over JET.
        return cv2.applyColorMap(heat_u8, cv2.COLORMAP_VIRIDIS)

    def _blend_heat_overlay(self, img: np.ndarray, heat_overlay: np.ndarray) -> np.ndarray:
        if heat_overlay.size == 0:
            return img
        mask = heat_overlay.max(axis=2) > 0
        if not np.any(mask):
            return img
        mixed = cv2.addWeighted(img, 1.0 - self.alpha, heat_overlay, self.alpha, 0.0)
        img[mask] = mixed[mask]
        return img

    def _normalize_heat_map_u8(self, heat_map: np.ndarray) -> np.ndarray:
        if heat_map.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        max_val = float(heat_map.max())
        if max_val <= 0.0:
            return np.zeros_like(heat_map, dtype=np.uint8)
        clip_val = float(np.percentile(heat_map, self.norm_percentile))
        clip_val = max(clip_val, max_val * 1e-6)
        heat = np.clip(heat_map / clip_val, 0.0, 1.0)
        return np.round(heat * 255.0).astype(np.uint8)

    def _draw_prev_track_bbox(self, img: np.ndarray, track_result: np.ndarray):
        if not isinstance(track_result, np.ndarray) or track_result.size == 0:
            return
        overlay = img.copy()
        for i in range(track_result.shape[0]):
            bb_left = int(track_result[i, 2])
            bb_top = int(track_result[i, 3])
            bb_width = int(track_result[i, 4])
            bb_height = int(track_result[i, 5])
            bb_right = bb_left + bb_width
            bb_bottom = bb_top + bb_height
            cv2.rectangle(
                overlay,
                (bb_left, bb_top),
                (bb_right, bb_bottom),
                (0, 255, 0),
                self.prev_track_bbox_thickness,
            )
        # Use a stronger alpha so previous-frame tracked regions read as stable context without textual clutter.
        cv2.addWeighted(overlay, self.prev_track_bbox_alpha, img, 1.0 - self.prev_track_bbox_alpha, 0.0, dst=img)

    def _draw_missing_gt_bbox(self, img: np.ndarray, boxes_xyxy: np.ndarray):
        for box in boxes_xyxy:
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            cv2.rectangle(img, (x1, y1), (x2, y2), (32, 32, 255), self.missing_gt_bbox_thickness)

    @staticmethod
    def _track_boxes_to_xyxy(track_result: np.ndarray) -> np.ndarray:
        if not isinstance(track_result, np.ndarray) or track_result.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        boxes = np.asarray(track_result[:, 2:6], dtype=np.float32)
        xyxy = np.zeros((boxes.shape[0], 4), dtype=np.float32)
        xyxy[:, 0] = boxes[:, 0]
        xyxy[:, 1] = boxes[:, 1]
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
        keep = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        return xyxy[keep]

    @staticmethod
    def _box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        if boxes1.size == 0 or boxes2.size == 0:
            return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
        lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = np.clip(rb - lt, a_min=0.0, a_max=None)
        inter = wh[..., 0] * wh[..., 1]
        area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], a_min=0.0, a_max=None) * np.clip(
            boxes1[:, 3] - boxes1[:, 1], a_min=0.0, a_max=None
        )
        area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], a_min=0.0, a_max=None) * np.clip(
            boxes2[:, 3] - boxes2[:, 1], a_min=0.0, a_max=None
        )
        union = np.clip(area1[:, None] + area2[None, :] - inter, a_min=1e-6, a_max=None)
        return inter / union

    def _append_video_bundle(
        self,
        frame: FrameContext,
        focus_data: dict[str, np.ndarray | int],
        prev_track_result: np.ndarray,
        focus_gt_boxes: np.ndarray,
        all_gt_boxes: np.ndarray,
    ):
        frame_name = self.frame_filename(frame, suffix="")
        heat_map_u8 = self._normalize_heat_map_u8(np.asarray(focus_data["heat_map"], dtype=np.float32))
        frame_record = {
            "stem": frame_name,
            "frame_id": int(frame.frame_id),
            "image_path": str(frame.img_path),
            "num_valid_queries": int(focus_data["num_valid_queries"]),
            "prev_track_count": int(prev_track_result.shape[0]),
            "all_gt_count": int(all_gt_boxes.shape[0]),
            "focus_gt_count": int(focus_gt_boxes.shape[0]),
            "all_gt_boxes": np.asarray(all_gt_boxes, dtype=np.float32).round(3).tolist(),
            "focus_gt_boxes": np.asarray(focus_gt_boxes, dtype=np.float32).round(3).tolist(),
            "heat_stats": {
                "max": float(np.asarray(focus_data["heat_map"], dtype=np.float32).max())
                if np.asarray(focus_data["heat_map"]).size > 0
                else 0.0,
                "sum": float(np.asarray(focus_data["heat_map"], dtype=np.float32).sum())
                if np.asarray(focus_data["heat_map"]).size > 0
                else 0.0,
            },
        }
        self._bundle_writer.submit(
            _BundleFrameJob(
                video_name=str(frame.video_name),
                bundle_path=os.path.join(self.task_dir(frame.video_name), self.BUNDLE_FILENAME),
                frame_record=frame_record,
                heat_map_u8=heat_map_u8,
            )
        )
