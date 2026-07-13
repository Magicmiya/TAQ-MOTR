from __future__ import annotations

import cv2
import numpy as np

from ..core import BaseVisualTask, FrameContext, HookEvent


class BBoxRenderTask(BaseVisualTask):
    def __init__(self, task_name: str, cfg: dict, mode: str, root_dir: str):
        super().__init__(task_name, cfg, mode, root_dir)
        self.save_image = bool(cfg.get("save_image", True))
        self.show_image = bool(cfg.get("show_image", False))
        self.window_delay = int(cfg.get("window_delay", 1))
        self.draw_frame_text = bool(cfg.get("draw_frame_text", True))

    def requires_image(self) -> bool:
        return bool(self.enabled and (self.save_image or self.show_image))

    def update(self, frame: FrameContext, hook_events: list[HookEvent]):
        del hook_events
        if not self.enabled:
            return
        img = frame.get_image_bgr()
        if img is None:
            return

        out_img = self._draw_bbox(np.ascontiguousarray(img.copy()), frame.track_result, frame.frame_id)
        if self.save_image:
            self.submit_image(frame, out_img)

        if self.show_image:
            cv2.imshow(frame.video_name, out_img)
            cv2.waitKey(self.window_delay)

    def close(self):
        super().close()
        if self.show_image:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _draw_bbox(self, img: np.ndarray, track_result: np.ndarray, frame_id: int) -> np.ndarray:
        if self.draw_frame_text:
            cv2.putText(
                img,
                f"Frame: {frame_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        if not isinstance(track_result, np.ndarray) or track_result.size == 0:
            return img

        for i in range(track_result.shape[0]):
            track_id = int(track_result[i, 1])
            bb_left = int(track_result[i, 2])
            bb_top = int(track_result[i, 3])
            bb_width = int(track_result[i, 4])
            bb_height = int(track_result[i, 5])
            bb_right = bb_left + bb_width
            bb_bottom = bb_top + bb_height

            cv2.rectangle(img, (bb_left, bb_top), (bb_right, bb_bottom), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"ID: {track_id}",
                (bb_left, bb_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return img
