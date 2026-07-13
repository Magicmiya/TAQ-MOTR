from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from ..core import BaseVisualTask, FrameContext, HookEvent


class HQGTopKRoiMapTask(BaseVisualTask):
    HOOK_HQG = "hqg_topk_source"

    def __init__(self, task_name: str, cfg: dict, mode: str, root_dir: str):
        super().__init__(task_name, cfg, mode, root_dir)
        self.save_image = bool(cfg.get("save_image", True))
        self.show_image = bool(cfg.get("show_image", False))
        self.window_delay = int(cfg.get("window_delay", 1))
        self.alpha = float(np.clip(float(cfg.get("alpha", 0.45)), 0.0, 1.0))
        self.min_score = float(cfg.get("min_score", 0.0))
        self.draw_frame_text = bool(cfg.get("draw_frame_text", True))
        self.draw_track_bbox = bool(cfg.get("draw_track_bbox", True))
        self.draw_query_bbox = bool(cfg.get("draw_query_bbox", False))
        self.track_bbox_alpha = float(np.clip(float(cfg.get("track_bbox_alpha", 0.28)), 0.0, 1.0))

    def required_switches(self) -> set[str]:
        return {"hqg_topk_source"}

    def requires_image(self) -> bool:
        return bool(self.enabled and (self.save_image or self.show_image))

    def update(self, frame: FrameContext, hook_events: list[HookEvent]):
        if not self.enabled:
            return

        hqg_event = self._latest_event(hook_events, self.HOOK_HQG)
        if hqg_event is None:
            return

        img = frame.get_image_bgr()
        if img is None:
            return

        roi_boxes, roi_scores = self._extract_topk_boxes(hqg_event.payload, img.shape[:2])
        out_img = np.ascontiguousarray(img.copy())
        if roi_boxes.shape[0] > 0:
            heat_overlay = self._build_heat_overlay(img.shape[:2], roi_boxes, roi_scores)
            out_img = self._blend_heat_overlay(out_img, heat_overlay)
            if self.draw_query_bbox:
                self._draw_query_boxes(out_img, roi_boxes)

        if self.draw_track_bbox:
            self._draw_track_bbox(out_img, frame.track_result)

        if self.draw_frame_text:
            # Show the actual number of decoder input queries rendered for this frame instead of a configurable top-k.
            cv2.putText(
                out_img,
                f"Frame: {frame.frame_id}  Decoder queries: {roi_boxes.shape[0]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        if self.save_image:
            self.submit_image(frame, out_img)

        if self.show_image:
            window_name = f"{frame.video_name}:{self.task_name}"
            cv2.imshow(window_name, out_img)
            cv2.waitKey(self.window_delay)

    def close(self):
        super().close()
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

    def _extract_topk_boxes(self, payload: dict, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        main_boxes = payload.get("main_boxes", None)
        main_logits = payload.get("main_logits", None)
        main_mask = payload.get("main_mask", None)
        if not isinstance(main_boxes, torch.Tensor) or not isinstance(main_logits, torch.Tensor):
            return np.zeros((0, 4), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        boxes = main_boxes.detach().to(device="cpu", dtype=torch.float32)
        logits = main_logits.detach().to(device="cpu", dtype=torch.float32)
        if boxes.ndim != 3 or logits.ndim != 3 or boxes.shape[0] == 0 or logits.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        boxes = boxes[0]
        scores = torch.sigmoid(logits[0]).amax(dim=-1)

        if isinstance(main_mask, torch.Tensor) and main_mask.ndim >= 2 and main_mask.shape[0] > 0:
            valid_mask = ~main_mask[0].detach().to(device="cpu", dtype=torch.bool)
        else:
            valid_mask = torch.ones((boxes.shape[0],), dtype=torch.bool)

        if valid_mask.numel() != boxes.shape[0]:
            valid_mask = torch.ones((boxes.shape[0],), dtype=torch.bool)

        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        if boxes.numel() == 0 or scores.numel() == 0:
            return np.zeros((0, 4), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        if self.min_score > 0.0:
            keep = scores >= self.min_score
            boxes = boxes[keep]
            scores = scores[keep]
        if boxes.numel() == 0 or scores.numel() == 0:
            return np.zeros((0, 4), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        boxes_xyxy = self._cxcywh_to_xyxy(boxes, image_shape=image_shape)
        keep = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        return boxes_xyxy[keep], scores.numpy().astype(np.float32)[keep]

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
        x2 = torch.clamp(cx + bw * 0.5, min=0.0, max=float(w))
        y2 = torch.clamp(cy + bh * 0.5, min=0.0, max=float(h))
        return torch.stack([x1, y1, x2, y2], dim=1).round().to(dtype=torch.int32).numpy()

    def _build_heat_overlay(
        self,
        image_shape: tuple[int, int],
        roi_boxes: np.ndarray,
        roi_scores: np.ndarray,
    ) -> np.ndarray:
        h, w = image_shape
        heat = np.zeros((h, w), dtype=np.float32)
        # Accumulate score-weighted box coverage to visualize where HQG TopK queries initially focus.
        for box, score in zip(roi_boxes, roi_scores):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            heat[y1:y2, x1:x2] += max(float(score), 1e-6)

        max_val = float(heat.max())
        if max_val <= 0.0:
            return np.zeros((h, w, 3), dtype=np.uint8)

        heat = np.clip(heat / max_val, 0.0, 1.0)
        return self._blue_red_heatmap(heat)

    @staticmethod
    def _blue_red_heatmap(heat: np.ndarray) -> np.ndarray:
        heat = np.clip(heat, 0.0, 1.0).astype(np.float32)
        blue = np.round((1.0 - heat) * 255.0).astype(np.uint8)
        red = np.round(heat * 255.0).astype(np.uint8)
        green = np.zeros_like(red, dtype=np.uint8)
        return np.stack([blue, green, red], axis=-1)

    def _blend_heat_overlay(self, img: np.ndarray, heat_overlay: np.ndarray) -> np.ndarray:
        if heat_overlay.size == 0:
            return img
        mask = heat_overlay.max(axis=2) > 0
        if not np.any(mask):
            return img
        mixed = cv2.addWeighted(img, 1.0 - self.alpha, heat_overlay, self.alpha, 0.0)
        img[mask] = mixed[mask]
        return img

    @staticmethod
    def _draw_query_boxes(img: np.ndarray, roi_boxes: np.ndarray):
        for box in roi_boxes:
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            cv2.rectangle(img, (x1, y1), (x2, y2), (32, 32, 255), 1)

    def _draw_track_bbox(self, img: np.ndarray, track_result: np.ndarray):
        if not isinstance(track_result, np.ndarray) or track_result.size == 0:
            return
        overlay = img.copy()
        for i in range(track_result.shape[0]):
            track_id = int(track_result[i, 1])
            bb_left = int(track_result[i, 2])
            bb_top = int(track_result[i, 3])
            bb_width = int(track_result[i, 4])
            bb_height = int(track_result[i, 5])
            bb_right = bb_left + bb_width
            bb_bottom = bb_top + bb_height

            cv2.rectangle(overlay, (bb_left, bb_top), (bb_right, bb_bottom), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"ID: {track_id}",
                (bb_left, max(12, bb_top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        # Blend tracker boxes with low alpha so ROI heat stays visually dominant while preserving target identity.
        cv2.addWeighted(overlay, self.track_bbox_alpha, img, 1.0 - self.track_bbox_alpha, 0.0, dst=img)
