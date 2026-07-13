from __future__ import annotations

import csv
import os
import threading
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import torch

from ..core import BaseVisualTask, FrameContext, HookEvent


class DetRecoverMonitorTask(BaseVisualTask):
    HOOK_RECOVER = "det_recover_monitor"
    REJECT_REASON_MAP = {
        0: "accepted",
        1: "threshold",
        2: "assignment",
    }

    def __init__(self, task_name: str, cfg: dict, mode: str, root_dir: str):
        super().__init__(task_name, cfg, mode, root_dir)
        self.save_image = bool(cfg.get("save_image", False))
        self.show_image = bool(cfg.get("show_image", False))
        self.window_delay = int(cfg.get("window_delay", 1))
        self.draw_frame_text = bool(cfg.get("draw_frame_text", True))
        self.save_csv = bool(cfg.get("save_csv", True))
        self.save_png = bool(cfg.get("save_png", True))
        self.save_all_frames = bool(cfg.get("save_all_frames", False))
        self.draw_rejected = bool(cfg.get("draw_rejected", False))
        self._rows_by_video: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
        self._global_rows: list[dict[str, float | int | str]] = []
        self._lock = threading.Lock()

    def required_switches(self) -> set[str]:
        return {self.HOOK_RECOVER}

    def requires_image(self) -> bool:
        return bool(self.enabled and (self.save_image or self.show_image))

    def update(self, frame: FrameContext, hook_events: list[HookEvent]):
        if not self.enabled:
            return
        recover_event = self._latest_event(hook_events, self.HOOK_RECOVER)
        if recover_event is None:
            return

        recover_vis = recover_event.payload.get("recover_vis", None)
        if not isinstance(recover_vis, list) or len(recover_vis) == 0:
            return

        rows: list[dict[str, float | int | str]] = []
        accepted_items: list[dict[str, float | int | np.ndarray]] = []
        rejected_items: list[dict[str, float | int | np.ndarray]] = []

        for batch_vis in recover_vis:
            if not isinstance(batch_vis, dict):
                continue
            lost_ids = self._to_numpy(batch_vis.get("lost_track_ids", None), dtype=np.int64, ndim=1)
            best_boxes = self._to_numpy(batch_vis.get("best_det_boxes", None), dtype=np.float32, ndim=2)
            best_det_indices = self._to_numpy(batch_vis.get("best_det_indices", None), dtype=np.int64, ndim=1)
            best_app_cos = self._to_numpy(batch_vis.get("best_app_cos", None), dtype=np.float32, ndim=1)
            best_motion_cost = self._to_numpy(batch_vis.get("best_motion_cost", None), dtype=np.float32, ndim=1)
            best_total_cost = self._to_numpy(batch_vis.get("best_total_cost", None), dtype=np.float32, ndim=1)
            best_age = self._to_numpy(batch_vis.get("best_age", None), dtype=np.int64, ndim=1)
            best_accepted = self._to_numpy(batch_vis.get("best_accepted", None), dtype=np.bool_, ndim=1)
            best_reject_reason = self._to_numpy(batch_vis.get("best_reject_reason", None), dtype=np.int64, ndim=1)

            if (
                lost_ids is None
                or best_boxes is None
                or best_det_indices is None
                or best_app_cos is None
                or best_motion_cost is None
                or best_total_cost is None
                or best_age is None
                or best_accepted is None
                or best_reject_reason is None
            ):
                continue

            n = min(
                len(lost_ids),
                len(best_boxes),
                len(best_det_indices),
                len(best_app_cos),
                len(best_motion_cost),
                len(best_total_cost),
                len(best_age),
                len(best_accepted),
                len(best_reject_reason),
            )
            for i in range(n):
                reason_code = int(best_reject_reason[i])
                row = {
                    "video_name": str(frame.video_name),
                    "frame_id": int(frame.frame_id),
                    "track_id": int(lost_ids[i]),
                    "best_det_idx": int(best_det_indices[i]),
                    "accepted": int(bool(best_accepted[i])),
                    "reject_reason": self.REJECT_REASON_MAP.get(reason_code, f"unknown_{reason_code}"),
                    "age": int(best_age[i]),
                    "app_cos": float(best_app_cos[i]),
                    "motion_cost": float(best_motion_cost[i]),
                    "total_cost": float(best_total_cost[i]),
                }
                rows.append(row)

                item = dict(row)
                item["box"] = best_boxes[i]
                if bool(best_accepted[i]):
                    accepted_items.append(item)
                else:
                    rejected_items.append(item)

        if rows:
            with self._lock:
                self._rows_by_video[frame.video_name].extend(rows)
                self._global_rows.extend(dict(row) for row in rows)

        if not (self.save_image or self.show_image):
            return
        if len(accepted_items) == 0 and not (self.save_all_frames or self.draw_rejected):
            return

        img = frame.get_image_bgr()
        if img is None:
            return

        out_img = np.ascontiguousarray(img.copy())
        if self.draw_frame_text:
            summary = f"Frame {frame.frame_id} | recover={len(accepted_items)}"
            cv2.putText(out_img, summary, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 255), 2)

        for item in accepted_items:
            self._draw_box(
                img=out_img,
                box=item["box"],
                image_shape=out_img.shape,
                color=(30, 30, 255),
                text=(
                    f"REC {int(item['track_id'])} "
                    f"cos={float(item['app_cos']):.2f} "
                    f"m={float(item['motion_cost']):.2f} "
                    f"c={float(item['total_cost']):.2f}"
                ),
            )

        if self.draw_rejected:
            for item in rejected_items:
                self._draw_box(
                    img=out_img,
                    box=item["box"],
                    image_shape=out_img.shape,
                    color=(0, 180, 255),
                    text=(
                        f"REJ {int(item['track_id'])} "
                        f"{item['reject_reason']} "
                        f"cos={float(item['app_cos']):.2f} "
                        f"c={float(item['total_cost']):.2f}"
                    ),
                    thickness=1,
                )

        if self.save_image:
            self.submit_image(frame, out_img)

        if self.show_image:
            cv2.imshow(frame.video_name + "_" + self.task_name, out_img)
            cv2.waitKey(self.window_delay)

    def close(self):
        if self.show_image:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if not (self.save_csv or self.save_png):
            return

        with self._lock:
            rows_by_video = {video_name: list(rows) for video_name, rows in self._rows_by_video.items()}
            global_rows = list(self._global_rows)

        for video_name, rows in rows_by_video.items():
            self._export_summary(
                out_dir=self.video_dir(video_name),
                rows=rows,
                title=f"Detection Recover Summary ({video_name})",
            )
        self._export_summary(out_dir=self.root_dir, rows=global_rows, title="Detection Recover Summary (All Videos)")

    def _export_summary(self, out_dir: str, rows: list[dict[str, float | int]], title: str):
        os.makedirs(out_dir, exist_ok=True)
        if self.save_csv:
            csv_path = os.path.join(out_dir, "det_recover_summary.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "video_name",
                        "frame_id",
                        "track_id",
                        "best_det_idx",
                        "accepted",
                        "reject_reason",
                        "age",
                        "app_cos",
                        "motion_cost",
                        "total_cost",
                    ],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        if self.save_png:
            self._save_hist_png(out_dir=out_dir, rows=rows, title=title)

    def _save_hist_png(self, out_dir: str, rows: list[dict[str, float | int]], title: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        accepted = np.array([float(r["app_cos"]) for r in rows if int(r["accepted"]) == 1], dtype=np.float32)
        rejected = np.array([float(r["app_cos"]) for r in rows if int(r["accepted"]) == 0], dtype=np.float32)
        accepted_cost = np.array([float(r["total_cost"]) for r in rows if int(r["accepted"]) == 1], dtype=np.float32)
        rejected_cost = np.array([float(r["total_cost"]) for r in rows if int(r["accepted"]) == 0], dtype=np.float32)
        accepted_motion = np.array([float(r["motion_cost"]) for r in rows if int(r["accepted"]) == 1], dtype=np.float32)
        rejected_motion = np.array([float(r["motion_cost"]) for r in rows if int(r["accepted"]) == 0], dtype=np.float32)
        accepted_age = np.array([float(r["age"]) for r in rows if int(r["accepted"]) == 1], dtype=np.float32)
        rejected_age = np.array([float(r["age"]) for r in rows if int(r["accepted"]) == 0], dtype=np.float32)
        threshold_rejects = sum(1 for r in rows if r.get("reject_reason") == "threshold")
        assignment_rejects = sum(1 for r in rows if r.get("reject_reason") == "assignment")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.reshape(-1)
        bins_cos = np.linspace(-1.0, 1.0, 41)
        bins_cost = np.linspace(0.0, 1.0, 41)
        bins_age = np.arange(0.0, max(float(accepted_age.max(initial=0.0)), float(rejected_age.max(initial=0.0)), 1.0) + 2.0)
        axes[0].hist(rejected, bins=bins_cos, alpha=0.55, label="rejected", color="#ffb347")
        axes[0].hist(accepted, bins=bins_cos, alpha=0.65, label="accepted", color="#d62728")
        axes[0].set_title("Appearance Cosine")
        axes[0].set_xlabel("cosine similarity")
        axes[0].set_ylabel("count")
        axes[0].legend()

        axes[1].hist(rejected_cost, bins=bins_cost, alpha=0.55, label="rejected", color="#ffb347")
        axes[1].hist(accepted_cost, bins=bins_cost, alpha=0.65, label="accepted", color="#1f77b4")
        axes[1].set_title("Recover Total Cost")
        axes[1].set_xlabel("cost")
        axes[1].set_ylabel("count")
        axes[1].legend()

        axes[2].hist(rejected_motion, bins=bins_cost, alpha=0.55, label="rejected", color="#ffb347")
        axes[2].hist(accepted_motion, bins=bins_cost, alpha=0.65, label="accepted", color="#2ca02c")
        axes[2].set_title("Motion Cost")
        axes[2].set_xlabel("cost")
        axes[2].set_ylabel("count")
        axes[2].legend()

        axes[3].hist(rejected_age, bins=bins_age, alpha=0.55, label="rejected", color="#ffb347")
        axes[3].hist(accepted_age, bins=bins_age, alpha=0.65, label="accepted", color="#9467bd")
        axes[3].set_title("Lost Age")
        axes[3].set_xlabel("frames")
        axes[3].set_ylabel("count")
        axes[3].legend()

        accepted_count = int(accepted.shape[0])
        rejected_count = int(rejected.shape[0])
        summary_lines = [
            f"accepted={accepted_count}, rejected={rejected_count}",
            f"rej_threshold={threshold_rejects}, rej_assignment={assignment_rejects}",
            f"acc_cos_mean={accepted.mean():.3f}" if accepted_count > 0 else "acc_cos_mean=n/a",
            f"rej_cos_mean={rejected.mean():.3f}" if rejected_count > 0 else "rej_cos_mean=n/a",
            f"acc_cost_mean={accepted_cost.mean():.3f}" if accepted_count > 0 else "acc_cost_mean=n/a",
            f"rej_cost_mean={rejected_cost.mean():.3f}" if rejected_count > 0 else "rej_cost_mean=n/a",
            f"acc_motion_mean={accepted_motion.mean():.3f}" if accepted_count > 0 else "acc_motion_mean=n/a",
            f"rej_motion_mean={rejected_motion.mean():.3f}" if rejected_count > 0 else "rej_motion_mean=n/a",
            f"acc_age_mean={accepted_age.mean():.2f}" if accepted_count > 0 else "acc_age_mean=n/a",
            f"rej_age_mean={rejected_age.mean():.2f}" if rejected_count > 0 else "rej_age_mean=n/a",
        ]
        fig.text(0.5, 0.01, " | ".join(summary_lines), ha="center", va="bottom", fontsize=9)
        fig.suptitle(title)
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.96])
        fig.savefig(os.path.join(out_dir, "det_recover_summary.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _latest_event(events: list[HookEvent], name: str) -> Optional[HookEvent]:
        for event in reversed(events):
            if event.name == name:
                return event
        return None

    @staticmethod
    def _to_numpy(value, dtype, ndim: int):
        if isinstance(value, torch.Tensor):
            arr = value.detach().to(device="cpu").numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            return None
        arr = arr.astype(dtype, copy=False)
        if arr.ndim != ndim:
            return None
        return arr

    @staticmethod
    def _draw_box(
        img: np.ndarray,
        box: np.ndarray,
        image_shape: tuple[int, ...],
        color: tuple[int, int, int],
        text: str,
        thickness: int = 2,
    ):
        h, w = image_shape[:2]
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw * 0.5) * w)
        y1 = int((cy - bh * 0.5) * h)
        x2 = int((cx + bw * 0.5) * w)
        y2 = int((cy + bh * 0.5) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            img,
            text,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
