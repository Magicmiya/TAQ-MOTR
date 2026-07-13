from __future__ import annotations

import os
import threading
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from ..core import BaseVisualTask, FrameContext, HookEvent


class HQGHistogramTask(BaseVisualTask):
    HOOK_HQG = "hqg_topk_source"
    HOOK_NEWBORN = "newborn_selector"

    def __init__(self, task_name: str, cfg: dict, mode: str, root_dir: str):
        super().__init__(task_name, cfg, mode, root_dir)
        bins = int(cfg.get("bins", 100))
        hist_range = cfg.get("range", [0.0, 1.0])
        lo = float(hist_range[0])
        hi = float(hist_range[1])
        self.bin_edges = np.linspace(lo, hi, bins + 1, dtype=np.float32)
        self.save_png = bool(cfg.get("save_png", True))
        self.save_csv = bool(cfg.get("save_csv", False))
        self.save_npz = bool(cfg.get("save_npz", False))

        self._hist_by_video: dict[str, dict[str, np.ndarray]] = {}
        self._frames_by_video: dict[str, int] = defaultdict(int)
        self._global_hist = self._new_hist()
        self._global_frames = 0
        self._lock = threading.Lock()

    def required_switches(self) -> set[str]:
        return {"hqg_topk_source", "newborn_selector"}

    def update(self, frame: FrameContext, hook_events: list[HookEvent]):
        if not self.enabled:
            return

        hqg_event = self._latest_event(hook_events, self.HOOK_HQG)
        newborn_event = self._latest_event(hook_events, self.HOOK_NEWBORN)
        if hqg_event is None or newborn_event is None:
            return

        main_logits = hqg_event.payload.get("main_logits", None)
        main_mask = hqg_event.payload.get("main_mask", None)
        pred_logits = newborn_event.payload.get("pred_logits", None)
        pred_mask = newborn_event.payload.get("pred_mask", None)
        high_conf_index = newborn_event.payload.get("high_conf_index", None)
        if (
            not isinstance(main_logits, torch.Tensor)
            or not isinstance(main_mask, torch.Tensor)
            or not isinstance(pred_logits, torch.Tensor)
            or not isinstance(pred_mask, torch.Tensor)
            or not isinstance(high_conf_index, list)
        ):
            return

        main_logits = main_logits.detach().to(device="cpu", dtype=torch.float32)
        main_mask = main_mask.detach().to(device="cpu", dtype=torch.bool)
        pred_logits = pred_logits.detach().to(device="cpu", dtype=torch.float32)
        pred_mask = pred_mask.detach().to(device="cpu", dtype=torch.bool)

        hqg_scores = torch.sigmoid(main_logits).max(dim=-1).values
        conf_scores = torch.sigmoid(pred_logits).max(dim=-1).values

        bsz = min(hqg_scores.shape[0], conf_scores.shape[0], len(high_conf_index))
        q_len = min(hqg_scores.shape[1], conf_scores.shape[1], main_mask.shape[1], pred_mask.shape[1])
        if bsz <= 0 or q_len <= 0:
            return

        hqg_scores = hqg_scores[:bsz, :q_len]
        conf_scores = conf_scores[:bsz, :q_len]
        valid_topk = (~main_mask[:bsz, :q_len].bool()) & (~pred_mask[:bsz, :q_len].bool())

        topk_hqg = hqg_scores[valid_topk]
        topk_conf = conf_scores[valid_topk]

        newborn_hqg_list = []
        newborn_conf_list = []
        for b in range(bsz):
            idx = high_conf_index[b]
            if not isinstance(idx, torch.Tensor) or idx.numel() == 0:
                continue
            idx = idx.detach().to(device="cpu", dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < q_len)]
            if idx.numel() == 0:
                continue
            newborn_hqg_list.append(hqg_scores[b].index_select(0, idx))
            newborn_conf_list.append(conf_scores[b].index_select(0, idx))

        newborn_hqg = (
            torch.cat(newborn_hqg_list, dim=0)
            if len(newborn_hqg_list) > 0
            else hqg_scores.new_zeros((0,), dtype=hqg_scores.dtype)
        )
        newborn_conf = (
            torch.cat(newborn_conf_list, dim=0)
            if len(newborn_conf_list) > 0
            else conf_scores.new_zeros((0,), dtype=conf_scores.dtype)
        )

        partial_hist = {
            "topk_hqg": self._histogram(topk_hqg),
            "topk_conf": self._histogram(topk_conf),
            "newborn_conf": self._histogram(newborn_conf),
            "newborn_hqg": self._histogram(newborn_hqg),
        }

        with self._lock:
            video_hist = self._hist_by_video.setdefault(frame.video_name, self._new_hist())
            for key, hist in partial_hist.items():
                if hist is None:
                    continue
                video_hist[key] += hist
                self._global_hist[key] += hist
            self._frames_by_video[frame.video_name] += 1
            self._global_frames += 1

    def close(self):
        if not self.enabled:
            return

        with self._lock:
            hist_by_video = {video_name: self._copy_hist(hist) for video_name, hist in self._hist_by_video.items()}
            frames_by_video = dict(self._frames_by_video)
            global_hist = self._copy_hist(self._global_hist)
            global_frames = int(self._global_frames)

        for video_name, hist in hist_by_video.items():
            out_dir = self.video_dir(video_name)
            self._export_summary(
                out_dir=out_dir,
                base_name="hqg_hist_summary",
                hist=hist,
                frame_count=frames_by_video.get(video_name, 0),
                video_count=1,
                title=f"HQG Histogram Summary ({video_name})",
            )

        self._export_summary(
            out_dir=self.root_dir,
            base_name="hqg_hist_summary",
            hist=global_hist,
            frame_count=global_frames,
            video_count=len(hist_by_video),
            title="HQG Histogram Summary (All Videos)",
        )

    def _histogram(self, values: torch.Tensor) -> Optional[np.ndarray]:
        if values.numel() == 0:
            return None
        np_values = values.detach().to(dtype=torch.float32, device="cpu").numpy()
        hist, _ = np.histogram(np_values, bins=self.bin_edges)
        return hist.astype(np.int64)

    def _latest_event(self, events: list[HookEvent], name: str) -> Optional[HookEvent]:
        for event in reversed(events):
            if event.name == name:
                return event
        return None

    def _new_hist(self) -> dict[str, np.ndarray]:
        return {
            "topk_hqg": np.zeros(len(self.bin_edges) - 1, dtype=np.int64),
            "topk_conf": np.zeros(len(self.bin_edges) - 1, dtype=np.int64),
            "newborn_conf": np.zeros(len(self.bin_edges) - 1, dtype=np.int64),
            "newborn_hqg": np.zeros(len(self.bin_edges) - 1, dtype=np.int64),
        }

    @staticmethod
    def _copy_hist(hist: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.array(values, copy=True) for key, values in hist.items()}

    def _export_summary(
        self,
        out_dir: str,
        base_name: str,
        hist: dict[str, np.ndarray],
        frame_count: int,
        video_count: int,
        title: str,
    ):
        os.makedirs(out_dir, exist_ok=True)
        legacy_png = os.path.join(out_dir, f"{base_name}.png")
        if os.path.exists(legacy_png):
            os.remove(legacy_png)

        if self.save_png:
            self._save_pngs(out_dir=out_dir, hist=hist, title=title)
        if self.save_csv:
            self._save_csv(os.path.join(out_dir, f"{base_name}.csv"), hist, frame_count, video_count)
        if self.save_npz:
            self._save_npz(os.path.join(out_dir, f"{base_name}.npz"), hist, frame_count, video_count)

    def _save_npz(self, out_file: str, hist: dict[str, np.ndarray], frame_count: int, video_count: int):
        np.savez_compressed(
            out_file,
            bin_edges=self.bin_edges,
            topk_hqg_hist=hist.get("topk_hqg", np.zeros(0, dtype=np.int64)),
            topk_conf_hist=hist.get("topk_conf", np.zeros(0, dtype=np.int64)),
            newborn_conf_hist=hist.get("newborn_conf", np.zeros(0, dtype=np.int64)),
            newborn_hqg_hist=hist.get("newborn_hqg", np.zeros(0, dtype=np.int64)),
            frame_count=np.array([frame_count], dtype=np.int64),
            video_count=np.array([video_count], dtype=np.int64),
            task_name=np.array([self.task_name]),
        )

    def _save_csv(self, out_file: str, hist: dict[str, np.ndarray], frame_count: int, video_count: int):
        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) * 0.5
        table = np.stack(
            [
                centers,
                hist["topk_hqg"].astype(np.float64),
                hist["topk_conf"].astype(np.float64),
                hist["newborn_hqg"].astype(np.float64),
                hist["newborn_conf"].astype(np.float64),
            ],
            axis=1,
        )
        header = (
            f"frame_count={frame_count},video_count={video_count}\n"
            "bin_center,topk_hqg,topk_conf,newborn_hqg,newborn_conf"
        )
        np.savetxt(out_file, table, delimiter=",", header=header, comments="")

    def _save_pngs(self, out_dir: str, hist: dict[str, np.ndarray], title: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        coarse = self._aggregate_hist_by_interval(hist=hist, interval=0.1)

        self._save_split_png(
            plt_mod=plt,
            out_file=os.path.join(out_dir, "hqg_hist_topk_summary.png"),
            title=f"{title} - TopK",
            a_values=coarse["topk_hqg"],
            b_values=coarse["topk_conf"],
            a_label="TopK HQG score",
            b_label="TopK confidence score",
            interval=0.1,
        )
        self._save_split_png(
            plt_mod=plt,
            out_file=os.path.join(out_dir, "hqg_hist_newborn_summary.png"),
            title=f"{title} - Newborn",
            a_values=coarse["newborn_hqg"],
            b_values=coarse["newborn_conf"],
            a_label="Newborn HQG score",
            b_label="Newborn confidence score",
            interval=0.1,
        )

    def _aggregate_hist_by_interval(self, hist: dict[str, np.ndarray], interval: float) -> dict[str, np.ndarray]:
        if interval <= 0:
            raise ValueError(f"interval must be > 0, got {interval}")
        coarse_edges = np.arange(0.0, 1.0 + 1e-8, interval, dtype=np.float32)
        if coarse_edges[-1] < 1.0:
            coarse_edges = np.append(coarse_edges, np.float32(1.0))

        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) * 0.5
        res: dict[str, np.ndarray] = {}
        for key, values in hist.items():
            out = np.zeros(len(coarse_edges) - 1, dtype=np.int64)
            for i in range(len(out)):
                lo = coarse_edges[i]
                hi = coarse_edges[i + 1]
                if i == len(out) - 1:
                    mask = (centers >= lo) & (centers <= hi)
                else:
                    mask = (centers >= lo) & (centers < hi)
                out[i] = values[mask].sum()
            res[key] = out
        return res

    @staticmethod
    def _save_split_png(
        plt_mod,
        out_file: str,
        title: str,
        a_values: np.ndarray,
        b_values: np.ndarray,
        a_label: str,
        b_label: str,
        interval: float,
    ):
        bins = np.arange(len(a_values), dtype=np.float32)
        labels = [f"[{i * interval:.1f}, {(i + 1) * interval:.1f})" for i in range(len(a_values))]
        if labels:
            labels[-1] = labels[-1].replace(")", "]")

        fig, ax = plt_mod.subplots(figsize=(12, 5))
        width = 0.38
        ax.bar(bins - width * 0.5, a_values, width=width, label=a_label, color="#1f77b4", alpha=0.85)
        ax.bar(bins + width * 0.5, b_values, width=width, label=b_label, color="#ff7f0e", alpha=0.80)
        ax.set_title(title)
        ax.set_xlabel("score interval")
        ax.set_ylabel("count")
        ax.set_xticks(bins)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_file, dpi=160, bbox_inches="tight")
        plt_mod.close(fig)
