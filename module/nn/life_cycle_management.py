# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)

import inspect
from enum import IntEnum
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .common import logits_to_scores
from .instance import TrackInstances as Track
from utils.visualizer import TensorHook


class TrackState(IntEnum):
    # State participation policy:
    # State     | QueryUpdater | Main track query | HQG inter memory | Lifecycle role
    # NEW       | update       | yes              | yes              | newborn or high-conf GT match
    # RELIABLE  | update       | yes              | yes              | stable tracked target
    # RECOVER   | no update    | yes              | yes              | weak positive from low-conf GT match
    # CONFUSED  | no update    | yes              | no               | reserved transition state
    # LOST      | no update    | yes              | no               | re-detection candidate
    # DEAD      | no           | no               | no               | LM-internal terminal, dropped before output
    # FAKE      | no           | shape guard      | no               | one-frame training shape guard
    # NOISE     | update       | yes              | no               | one-frame hard negative
    NEW = 0
    RELIABLE = 1
    RECOVER = 2  # only for training: GT-matched low-confidence detection
    CONFUSED = 3
    LOST = 4
    DEAD = 5
    FAKE = 6  # only for training
    NOISE = 7  # only for training


class LifeCycleManagement(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        high_conf_threshold: float,
        new_born_threshold: float,
        track_thresh: float,
        miss_tolerance: int,
        long_memory_lambda: float,
        sudden_death_threshold: float = 0.5,
        tp_drop_ratio: float = 0.0,
        fp_insert_ratio: float = 0.0,
        no_tracking_augment: bool = True,
        recover_iou_threshold: float = 0.5,
        iou_conf_threshold: float = 0.5,
        det_recover_enable: bool = True,
        det_recover_max_time: int = 8,
        det_recover_min_history: int = 2,
        det_recover_app_weight: float = 0.55,
        det_recover_motion_weight: float = 0.45,
        det_recover_cost_threshold: float = 0.5,
        det_recover_center_sigma: float = 0.35,
        det_recover_shape_sigma: float = 0.12,
        visualize: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.high_conf_threshold = high_conf_threshold
        self.new_born_threshold = new_born_threshold
        self.track_thresh = track_thresh
        self.miss_tolerance = miss_tolerance
        self.sudden_death_threshold = sudden_death_threshold
        self.long_memory_lambda = long_memory_lambda
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
        self.no_tracking_augment = no_tracking_augment
        self.recover_iou_threshold = recover_iou_threshold
        self.iou_conf_threshold = iou_conf_threshold
        self.det_recover_enable = det_recover_enable
        self.det_recover_max_time = det_recover_max_time
        self.det_recover_min_history = det_recover_min_history
        self.det_recover_app_weight = det_recover_app_weight
        self.det_recover_motion_weight = det_recover_motion_weight
        self.det_recover_cost_threshold = det_recover_cost_threshold
        self.det_recover_center_sigma = det_recover_center_sigma
        self.det_recover_shape_sigma = det_recover_shape_sigma
        self.visualize = visualize

        self.frame_idx = 0
        self.max_obj_id = 1

    @staticmethod
    def is_confused(hist_conf: torch.Tensor, last_conf: torch.Tensor) -> torch.Tensor:
        """
        Reserved hook for future CONFUSED-state logic.
        Current version returns all-False and keeps CONFUSED disabled.
        """
        del hist_conf
        return torch.zeros_like(last_conf, dtype=torch.bool)

    def init_a_clip(self, batch: Dict[str, Any], device: torch.device):
        del batch, device
        self.frame_idx = 0
        self.max_obj_id = 1

    def _next_ids(self, num: int, device: torch.device) -> torch.Tensor:
        if num <= 0:
            return torch.zeros((0,), dtype=torch.long, device=device)
        ids = torch.arange(self.max_obj_id, self.max_obj_id + num, dtype=torch.long, device=device)
        self.max_obj_id += num
        return ids

    def _make_history_kwargs(
        self,
        like_track: Track,
        boxes: torch.Tensor,
        output_embed: torch.Tensor,
        frame_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Create detached history tensors for freshly created tracks."""
        num = int(boxes.shape[0])
        history_len = like_track.history_len
        device = like_track.device

        hist_boxes = torch.zeros((num, history_len, 4), dtype=torch.float32, device=device)
        hist_frame_idx = torch.full((num, history_len), -1, dtype=torch.long, device=device)
        hist_output_embed = torch.zeros((num, history_len, like_track.hidden_dim), dtype=torch.float32, device=device)
        hist_ptr = torch.zeros((num,), dtype=torch.long, device=device)
        hist_count = torch.zeros((num,), dtype=torch.long, device=device)

        if num > 0:
            hist_boxes[:, 0, :] = boxes.detach().to(device=device, dtype=torch.float32)
            hist_frame_idx[:, 0] = int(frame_idx)
            hist_output_embed[:, 0, :] = output_embed.detach().to(device=device, dtype=torch.float32)
            hist_ptr[:] = 1 % history_len
            hist_count[:] = 1

        return {
            "hist_boxes": hist_boxes,
            "hist_frame_idx": hist_frame_idx,
            "hist_output_embed": hist_output_embed,
            "hist_ptr": hist_ptr,
            "hist_count": hist_count,
        }

    @staticmethod
    def _box_motion_state(boxes: torch.Tensor) -> torch.Tensor:
        wh = boxes[..., 2:].clamp_min(1e-6)
        return torch.cat([boxes[..., :2], torch.log(wh)], dim=-1)

    def _compute_recover_app_cost(
        self,
        track: Track,
        track_idx: int,
        det_query_embed: torch.Tensor,
    ) -> torch.Tensor:
        age = track.disappear_time[track_idx].to(dtype=torch.float32)
        max_time = max(int(self.det_recover_max_time), 1)
        alpha = torch.clamp(age / float(max_time), min=0.0, max=1.0)

        long_feat = F.normalize(track.long_memory[track_idx], dim=0, eps=1e-6)
        query_feat = F.normalize(track.query_embed[track_idx], dim=0, eps=1e-6)
        mixed_feat = F.normalize(alpha * long_feat + (1.0 - alpha) * query_feat, dim=0, eps=1e-6)
        det_feat = F.normalize(det_query_embed, dim=1, eps=1e-6)
        similarity = torch.matmul(det_feat, mixed_feat).clamp(-1.0, 1.0)
        return 0.5 * (1.0 - similarity)

    def _compute_recover_motion_cost(
        self,
        frames: torch.Tensor,
        boxes: torch.Tensor,
        det_boxes: torch.Tensor,
        current_frame: int,
    ) -> torch.Tensor:
        if frames.numel() == 0:
            return torch.full((det_boxes.shape[0],), 0.5, dtype=torch.float32, device=det_boxes.device)

        history_state = self._box_motion_state(boxes)
        last_frame = int(frames[-1].item())
        dt_cur = max(int(current_frame - last_frame), 1)

        if frames.numel() >= self.det_recover_min_history:
            hist_dt = (frames[1:] - frames[:-1]).to(dtype=torch.float32)
            valid = hist_dt > 0
            if valid.any():
                velocity = (history_state[1:] - history_state[:-1])[valid] / hist_dt[valid].unsqueeze(1)
                mean_velocity = velocity.mean(dim=0)
                var_velocity = velocity.var(dim=0, unbiased=False) if velocity.shape[0] > 1 else torch.zeros_like(mean_velocity)
            else:
                mean_velocity = torch.zeros((4,), dtype=torch.float32, device=det_boxes.device)
                var_velocity = torch.zeros((4,), dtype=torch.float32, device=det_boxes.device)
        else:
            mean_velocity = torch.zeros((4,), dtype=torch.float32, device=det_boxes.device)
            var_velocity = torch.zeros((4,), dtype=torch.float32, device=det_boxes.device)

        last_box = boxes[-1]
        last_state = history_state[-1]
        det_state = self._box_motion_state(det_boxes)
        observed_delta = det_state - last_state.unsqueeze(0)
        pred_mean = mean_velocity.unsqueeze(0) * float(dt_cur)

        box_scale = torch.sqrt((last_box[2] * last_box[3]).clamp_min(1e-4))
        center_std = self.det_recover_center_sigma * box_scale.clamp_min(0.05)
        shape_std = torch.full((2,), self.det_recover_shape_sigma, dtype=torch.float32, device=det_boxes.device)
        base_std = torch.cat(
            [
                center_std.repeat(2),
                shape_std,
            ],
            dim=0,
        )
        pred_var = var_velocity.unsqueeze(0) * float(dt_cur) + base_std.square().unsqueeze(0) * float(dt_cur)
        normalized_error = ((observed_delta - pred_mean) ** 2 / pred_var.clamp_min(1e-6)).mean(dim=1)
        return 1.0 - torch.exp(-0.5 * normalized_error)

    @staticmethod
    def _empty_recover_vis(device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "lost_track_ids": torch.zeros((0,), dtype=torch.long, device=device),
            "best_det_indices": torch.zeros((0,), dtype=torch.long, device=device),
            "best_det_boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
            "best_app_cos": torch.zeros((0,), dtype=torch.float32, device=device),
            "best_motion_cost": torch.zeros((0,), dtype=torch.float32, device=device),
            "best_total_cost": torch.zeros((0,), dtype=torch.float32, device=device),
            "best_age": torch.zeros((0,), dtype=torch.long, device=device),
            "best_accepted": torch.zeros((0,), dtype=torch.bool, device=device),
            "best_reject_reason": torch.zeros((0,), dtype=torch.long, device=device),
            "accepted_track_ids": torch.zeros((0,), dtype=torch.long, device=device),
            "accepted_det_boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
            "accepted_app_cos": torch.zeros((0,), dtype=torch.float32, device=device),
            "accepted_motion_cost": torch.zeros((0,), dtype=torch.float32, device=device),
            "accepted_total_cost": torch.zeros((0,), dtype=torch.float32, device=device),
            "accepted_age": torch.zeros((0,), dtype=torch.long, device=device),
        }

    @TensorHook(keys=["recover_vis"], name="det_recover_monitor", switch="det_recover_monitor")
    def _recover_lost_tracks_with_detection(
        self,
        tracks: List[Track],
        out: Dict[str, Any],
        high_conf_index: List[torch.Tensor],
    ) -> tuple[List[Track], List[torch.Tensor]]:
        """Recover LOST tracks before unmatched detections are converted into NEW tracks."""
        pred_logits = out["main_outputs"]["pred_logits"]
        pred_bboxes = out["main_outputs"]["pred_bboxes"]
        pred_queries = out["pred_queries"]
        pred_ref_points = out["pred_ref_point"]
        current_frame = self.frame_idx + 1
        remaining_indexes: List[torch.Tensor] = []
        recover_vis: List[Dict[str, torch.Tensor]] = []

        for b, (track, det_idx) in enumerate(zip(tracks, high_conf_index)):
            if len(track) == 0 or det_idx.numel() == 0:
                recover_vis.append(self._empty_recover_vis(device=track.device))
                remaining_indexes.append(det_idx)
                continue

            lost_mask = (track.states == int(TrackState.LOST)) & (track.disappear_time <= int(self.det_recover_max_time))
            lost_idx = torch.nonzero(lost_mask, as_tuple=False).squeeze(-1)
            if lost_idx.numel() == 0:
                recover_vis.append(self._empty_recover_vis(device=track.device))
                remaining_indexes.append(det_idx)
                continue

            det_idx = det_idx.to(device=track.device, dtype=torch.long)
            det_boxes = pred_bboxes[b, det_idx, :]
            det_query_embed = pred_queries[-2, b, det_idx, :]
            app_cost_matrix = torch.full(
                (lost_idx.numel(), det_idx.numel()),
                1.0,
                dtype=torch.float32,
                device=track.device,
            )
            motion_cost_matrix = torch.full_like(app_cost_matrix, 1.0)
            cost_matrix = torch.full(
                (lost_idx.numel(), det_idx.numel()),
                1.0,
                dtype=torch.float32,
                device=track.device,
            )
            track_ids = track.ids.index_select(0, lost_idx)
            track_age = track.disappear_time.index_select(0, lost_idx)

            for row, track_i in enumerate(lost_idx.detach().cpu().tolist()):
                hist_frames, hist_boxes = track.get_history(int(track_i))
                app_cost = self._compute_recover_app_cost(track=track, track_idx=int(track_i), det_query_embed=det_query_embed)
                motion_cost = self._compute_recover_motion_cost(
                    frames=hist_frames,
                    boxes=hist_boxes,
                    det_boxes=det_boxes,
                    current_frame=current_frame,
                )
                app_cost_matrix[row, :] = app_cost
                motion_cost_matrix[row, :] = motion_cost
                cost_matrix[row, :] = (
                    self.det_recover_app_weight * app_cost + self.det_recover_motion_weight * motion_cost
                )

            best_total_cost, best_cols = cost_matrix.min(dim=1)
            best_app_cost = app_cost_matrix.gather(1, best_cols.unsqueeze(1)).squeeze(1)
            best_motion_cost = motion_cost_matrix.gather(1, best_cols.unsqueeze(1)).squeeze(1)
            best_app_cos = (1.0 - 2.0 * best_app_cost).clamp(-1.0, 1.0)
            best_det_boxes = det_boxes.index_select(0, best_cols)
            best_accepted = torch.zeros((lost_idx.numel(),), dtype=torch.bool, device=track.device)
            best_reject_reason = torch.full((lost_idx.numel(),), 2, dtype=torch.long, device=track.device)

            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            keep_det = torch.ones((det_idx.numel(),), dtype=torch.bool, device=track.device)
            accepted_track_ids: list[int] = []
            accepted_det_boxes: list[torch.Tensor] = []
            accepted_app_cos: list[float] = []
            accepted_motion_cost: list[float] = []
            accepted_total_cost: list[float] = []
            accepted_age: list[int] = []
            for row, col in zip(row_ind.tolist(), col_ind.tolist()):
                if float(cost_matrix[row, col].item()) >= float(self.det_recover_cost_threshold):
                    best_reject_reason[row] = 1
                    continue

                track_i = int(lost_idx[row].item())
                det_i = int(det_idx[col].item())
                det_output = pred_queries[-1, b, det_i, :]
                best_accepted[row] = True
                best_reject_reason[row] = 0

                track.query_embed[track_i] = pred_queries[-2, b, det_i, :]
                track.ref_pts[track_i] = pred_ref_points[-2, b, det_i, :]
                track.boxes[track_i] = pred_bboxes[b, det_i, :]
                track.logits[track_i] = pred_logits[b, det_i, :]
                track.labels[track_i] = torch.argmax(pred_logits[b, det_i, :], dim=-1)
                track.output_embed[track_i] = det_output
                track.last_output[track_i] = det_output
                track.last_appear_boxes[track_i] = pred_bboxes[b, det_i, :]
                track.matched_idx[track_i] = det_i
                track.disappear_time[track_i] = 0
                track.states[track_i] = int(TrackState.RELIABLE)
                track.push_history_(
                    indices=torch.as_tensor([track_i], dtype=torch.long, device=track.device),
                    frame_idx=current_frame,
                    boxes=pred_bboxes[b, det_i, :].unsqueeze(0),
                    output_embed=det_output.unsqueeze(0),
                )
                keep_det[col] = False
                accepted_track_ids.append(int(track.ids[track_i].item()))
                accepted_det_boxes.append(pred_bboxes[b, det_i, :].detach())
                accepted_app_cos.append(float((1.0 - 2.0 * app_cost_matrix[row, col]).item()))
                accepted_motion_cost.append(float(motion_cost_matrix[row, col].item()))
                accepted_total_cost.append(float(cost_matrix[row, col].item()))
                accepted_age.append(int(track_age[row].item()))

            remaining_indexes.append(det_idx[keep_det])
            accepted_det_boxes_tensor = (
                torch.stack(accepted_det_boxes, dim=0)
                if len(accepted_det_boxes) > 0
                else torch.zeros((0, 4), dtype=torch.float32, device=track.device)
            )
            recover_vis.append(
                {
                    "lost_track_ids": track_ids,
                    "best_det_indices": det_idx.index_select(0, best_cols),
                    "best_det_boxes": best_det_boxes,
                    "best_app_cos": best_app_cos,
                    "best_motion_cost": best_motion_cost,
                    "best_total_cost": best_total_cost,
                    "best_age": track_age,
                    "best_accepted": best_accepted,
                    "best_reject_reason": best_reject_reason,
                    "accepted_track_ids": torch.as_tensor(accepted_track_ids, dtype=torch.long, device=track.device),
                    "accepted_det_boxes": accepted_det_boxes_tensor,
                    "accepted_app_cos": torch.as_tensor(accepted_app_cos, dtype=torch.float32, device=track.device),
                    "accepted_motion_cost": torch.as_tensor(
                        accepted_motion_cost, dtype=torch.float32, device=track.device
                    ),
                    "accepted_total_cost": torch.as_tensor(accepted_total_cost, dtype=torch.float32, device=track.device),
                    "accepted_age": torch.as_tensor(accepted_age, dtype=torch.long, device=track.device),
                }
            )

        return tracks, remaining_indexes

    @TensorHook(
        keys=["pred_logits", "pred_mask", "high_conf_index"], name="newborn_selector", switch="newborn_selector"
    )
    def _get_high_conf_det_idx(self, out: Dict[str, Any]) -> List[torch.Tensor]:
        _, det_num, _ = out["meta"]["query_nums"]
        main_out = out["main_outputs"]
        pred_logits = main_out["pred_logits"][:, :det_num, :]
        pred_mask = main_out.get("query_mask", None)
        if pred_mask is None:
            pred_mask = torch.zeros(pred_logits.shape[:2], dtype=torch.bool, device=pred_logits.device)
        else:
            pred_mask = pred_mask[:, :det_num]

        pred_scores = logits_to_scores(pred_logits)
        threshold = self.high_conf_threshold if self.training else self.new_born_threshold
        high_conf_mask = torch.max(pred_scores, dim=2).values >= threshold
        high_conf_mask = high_conf_mask & (~pred_mask)

        high_conf_index = [torch.where(mask)[0] for mask in high_conf_mask]
        return high_conf_index

    def _build_new_born_instances(
        self, out: Dict[str, Any], tracks: List[Track], indexes: List[torch.Tensor]
    ) -> List[Track]:
        device = out["pred_queries"].device
        pred_logits = out["main_outputs"]["pred_logits"]
        pred_bboxes = out["main_outputs"]["pred_bboxes"]
        pred_queries = out["pred_queries"]
        pred_ref_point = out["pred_ref_point"]
        current_frame = self.frame_idx + 1

        new_instances: List[Track] = []
        for b, idx in enumerate(indexes):
            _len = len(idx)
            if _len == 0:
                new_instances.append(Track.as_like(tracks[b]))
                continue

            ids = self._next_ids(_len, device=device)
            history_kwargs = self._make_history_kwargs(
                like_track=tracks[b],
                boxes=pred_bboxes[b, idx, :],
                output_embed=pred_queries[-1, b, idx, :],
                frame_idx=current_frame,
            )
            new_instances.append(
                Track.as_like(
                    tracks[b],
                    **{
                        "query_embed": pred_queries[-2, b, idx, :],
                        "ref_pts": pred_ref_point[-2, b, idx, :],
                        "ids": ids,
                        # Store detection-query index for recover grouping.
                        "matched_idx": idx.to(dtype=torch.long, device=device),
                        "boxes": pred_bboxes[b, idx, :],
                        "logits": pred_logits[b, idx, :],
                        "labels": torch.argmax(pred_logits[b, idx, :], dim=-1),
                        "output_embed": pred_queries[-1, b, idx, :],
                        "last_output": pred_queries[-1, b, idx, :],
                        "long_memory": pred_queries[-2, b, idx, :],
                        "last_appear_boxes": pred_bboxes[b, idx, :],
                        "disappear_time": torch.zeros(_len, dtype=torch.long, device=device),
                        "iou": torch.zeros(_len, dtype=torch.float32, device=device),
                        "states": torch.full(
                            (_len,),
                            int(TrackState.NEW),
                            dtype=torch.long,
                            device=device,
                        ),
                        **history_kwargs,
                    },
                )
            )
        return new_instances

    @staticmethod
    def _drop_transient_tracks(tracks: List[Track]) -> List[Track]:
        return [
            track
            if len(track) == 0
            else track[(track.states != int(TrackState.NOISE)) & (track.states != int(TrackState.FAKE))]
            for track in tracks
        ]

    def _select_active_tracks_train(self, previous_tracks: List[Track], new_tracks: List[Track]) -> List[Track]:
        tracks: List[Track] = []
        for prev_track, new_track in zip(previous_tracks, new_tracks):
            active_tracks = Track.cat(prev_track, new_track)

            if self.tp_drop_ratio == 0.0 and self.fp_insert_ratio == 0.0:
                if len(active_tracks) > 0:
                    # Match early QueryUpdater semantics: keep first, then mark low-IoU positives id=-1.
                    low_iou_pos = (active_tracks.ids >= 0) & (active_tracks.iou <= float(self.iou_conf_threshold))
                    active_tracks.ids[low_iou_pos] = -1
            else:
                keep_idxes = (active_tracks.iou >= float(self.iou_conf_threshold)) & (active_tracks.ids >= 0)
                active_tracks = active_tracks[keep_idxes]
                if self.tp_drop_ratio > 0.0 and not self.no_tracking_augment and len(active_tracks) > 0:
                    tp_keep_idx = torch.rand((len(active_tracks),), device=active_tracks.device) > self.tp_drop_ratio
                    active_tracks = active_tracks[tp_keep_idx]

            if len(active_tracks) == 0:
                device = prev_track.device
                active_tracks = Track.as_like(
                    prev_track,
                    **{
                        "query_embed": torch.randn(
                            (1, prev_track.hidden_dim),
                            dtype=torch.float32,
                            device=device,
                        ),
                        "ref_pts": torch.randn((1, 4), dtype=torch.float32, device=device),
                        "ids": torch.as_tensor([-2], dtype=torch.long, device=device),
                        "matched_idx": torch.as_tensor([-2], dtype=torch.long, device=device),
                        "last_appear_boxes": torch.randn((1, 4), dtype=torch.float32, device=device),
                        "boxes": torch.randn((1, 4), dtype=torch.float32, device=device),
                        "logits": torch.randn(
                            (1, prev_track.num_classes),
                            dtype=torch.float32,
                            device=device,
                        ),
                        "output_embed": torch.randn(
                            (1, prev_track.hidden_dim),
                            dtype=torch.float32,
                            device=device,
                        ),
                        "last_output": torch.randn(
                            (1, prev_track.hidden_dim),
                            dtype=torch.float32,
                            device=device,
                        ),
                        "long_memory": torch.randn(
                            (1, prev_track.hidden_dim),
                            dtype=torch.float32,
                            device=device,
                        ),
                        "disappear_time": torch.zeros(1, dtype=torch.long, device=device),
                        "labels": torch.zeros(1, dtype=torch.long, device=device),
                        "iou": torch.zeros(1, dtype=torch.float32, device=device),
                        "states": torch.full((1,), int(TrackState.FAKE), dtype=torch.long, device=device),
                    },
                )

            tracks.append(active_tracks)

        return tracks

    def _update_tracks_from_main_output(self, tracks: List[Track], out: Dict[str, Any]) -> List[Track]:
        """Update existing tracks with decoder outputs from the main branch.

        Args:
            tracks (List[Track]): Track instances carried from the previous frame.
                Each batch element is updated in place using the track-query part of
                the decoder output.
            out (Dict[str, Any]): Decoder output dictionary. This function reads the
                main branch predictions after the detection-query slice and treats
                them as aligned track-query predictions.

        Raises:
            ValueError: Raised when the number of valid track-query predictions does
                not match the number of track instances expected to be updated.

        Returns:
            List[Track]: The input track list after refreshing boxes, logits,
                output features, disappear counters, and lifecycle states
                (RELIABLE, LOST, or DEAD for non-transient tracks).
        """
        _, n_det, _ = out["meta"]["query_nums"]
        main_out = out["main_outputs"]

        for b, track in enumerate(tracks):
            track.frame = self.frame_idx + 1
            if len(track) == 0:
                continue

            if not self.training:
                track.last_appear_boxes = track.boxes.clone()
            prev_states = track.states.clone()
            pred_mask = main_out.get("query_mask", None)
            pred_mask_b = pred_mask[b, n_det:] if pred_mask is not None else None
            pred_bboxes = main_out["pred_bboxes"][b, n_det:, :]
            pred_logits = main_out["pred_logits"][b, n_det:, :]
            pred_queries = out["pred_queries"][:, b, n_det:, :]

            valid_idx = (
                torch.nonzero(~pred_mask_b, as_tuple=False).squeeze(-1)
                if pred_mask_b is not None
                else torch.arange(pred_logits.shape[0], device=track.device)
            )
            if len(valid_idx) != len(track):
                raise ValueError(
                    f"track valid queries {len(valid_idx)} != selected instances {len(track)} "
                    f"(all instances={len(track)})"
                )
            mapped_idx = valid_idx

            survive_mask = torch.zeros((len(track),), dtype=torch.bool, device=track.device)
            sudden_death_mask = torch.zeros((len(track),), dtype=torch.bool, device=track.device)
            if mapped_idx.numel() > 0:
                track.boxes[mapped_idx] = pred_bboxes[valid_idx]
                track.logits[mapped_idx] = pred_logits[valid_idx]
                track.output_embed[mapped_idx] = pred_queries[-1, valid_idx, :]

                scores = torch.max(logits_to_scores(track.logits[mapped_idx]), dim=1).values
                conf_mask = scores >= self.track_thresh
                if not self.training:
                    # Sudden death only applies in inference to tracks that were NEW in the previous frame.
                    prev_new_mask = prev_states[mapped_idx] == int(TrackState.NEW)
                    sudden_death_mask[mapped_idx] = prev_new_mask & (scores < self.sudden_death_threshold)
                new_labels = torch.argmax(track.logits[mapped_idx], dim=-1)
                label_changed = track.labels[mapped_idx] != new_labels
                track.labels[mapped_idx] = new_labels
                survive_mask[mapped_idx] = conf_mask & (~label_changed) & (~sudden_death_mask[mapped_idx])

            update_mask = torch.zeros((len(track),), dtype=torch.bool, device=track.device)
            if mapped_idx.numel() > 0:
                update_mask[mapped_idx] = True
            aging_mask = (~update_mask) & (track.states <= int(TrackState.LOST))
            fail_mask = update_mask & (~survive_mask) & (~sudden_death_mask)
            track.disappear_time[aging_mask | fail_mask] += 1
            track.disappear_time[survive_mask] = 0
            track.last_appear_boxes[survive_mask] = track.boxes[survive_mask]
            survive_idx = torch.nonzero(survive_mask, as_tuple=False).squeeze(-1)
            if survive_idx.numel() > 0:
                track.push_history_(
                    indices=survive_idx,
                    frame_idx=self.frame_idx + 1,
                    boxes=track.boxes.index_select(0, survive_idx),
                    output_embed=track.output_embed.index_select(0, survive_idx),
                )
            dead_mask = track.disappear_time >= self.miss_tolerance

            # Keep transient states isolated; they are removed after one-frame propagation.
            transient_mask = (track.states == int(TrackState.NOISE)) | (track.states == int(TrackState.FAKE))
            valid_state_mask = (~transient_mask) & (~sudden_death_mask)

            reliable_mask = survive_mask & (~dead_mask)
            lost_mask = (~reliable_mask) & (~dead_mask) & (~sudden_death_mask)
            track.states[valid_state_mask & reliable_mask] = int(TrackState.RELIABLE)
            track.states[valid_state_mask & lost_mask] = int(TrackState.LOST)
            track.states[valid_state_mask & dead_mask] = int(TrackState.DEAD)
            if sudden_death_mask.any():
                tracks[b] = track[~sudden_death_mask]

        return tracks

    def _build_training_det_instances(
        self,
        out: Dict[str, Any],
        tracks: List[Track],
        high_conf_index: List[torch.Tensor],
        det_matched: List[Dict[str, torch.Tensor]] | None,
    ) -> List[Track]:
        device = out["pred_queries"].device
        pred_logits = out["main_outputs"]["pred_logits"]
        pred_bboxes = out["main_outputs"]["pred_bboxes"]
        pred_queries = out["pred_queries"]
        pred_ref_point = out["pred_ref_point"]
        current_frame = self.frame_idx + 1
        new_instances: List[Track] = []

        for b, track in enumerate(tracks):
            high_idx = high_conf_index[b].to(device=device, dtype=torch.long)
            meta = det_matched[b] if det_matched is not None and b < len(det_matched) else {}
            matched_idx = meta.get("det_query_idx", torch.zeros((0,), dtype=torch.long, device=device)).to(
                device=device, dtype=torch.long
            )
            gt_ids = meta.get("gt_ids", torch.zeros((0,), dtype=torch.long, device=device)).to(device=device, dtype=torch.long)
            pair_iou = meta.get("iou", torch.zeros((0,), dtype=torch.float32, device=device)).to(
                device=device, dtype=torch.float32
            )

            if matched_idx.numel() > 0:
                high_matched = torch.isin(matched_idx, high_idx) if high_idx.numel() > 0 else torch.zeros_like(matched_idx, dtype=torch.bool)
                matched_states = torch.where(
                    high_matched,
                    torch.full_like(matched_idx, int(TrackState.NEW)),
                    torch.full_like(matched_idx, int(TrackState.RECOVER)),
                )
            else:
                matched_states = torch.zeros((0,), dtype=torch.long, device=device)

            if high_idx.numel() > 0 and matched_idx.numel() > 0:
                unmatched_high = high_idx[~torch.isin(high_idx, matched_idx)]
            else:
                unmatched_high = high_idx
            noise_len = int(unmatched_high.numel())
            noise_ids = torch.full((noise_len,), -1, dtype=torch.long, device=device)
            noise_states = torch.full((noise_len,), int(TrackState.NOISE), dtype=torch.long, device=device)
            noise_iou = torch.zeros((noise_len,), dtype=torch.float32, device=device)

            idx = torch.cat([matched_idx, unmatched_high], dim=0)
            if idx.numel() == 0:
                new_instances.append(Track.as_like(track))
                continue

            ids = torch.cat([gt_ids, noise_ids], dim=0)
            states = torch.cat([matched_states, noise_states], dim=0)
            iou = torch.cat([pair_iou, noise_iou], dim=0)
            history_kwargs = self._make_history_kwargs(
                like_track=track,
                boxes=pred_bboxes[b, idx, :],
                output_embed=pred_queries[-1, b, idx, :],
                frame_idx=current_frame,
            )
            new_instances.append(
                Track.as_like(
                    track,
                    **{
                        "query_embed": pred_queries[-2, b, idx, :],
                        "ref_pts": pred_ref_point[-2, b, idx, :],
                        "ids": ids,
                        "matched_idx": idx,
                        "boxes": pred_bboxes[b, idx, :],
                        "logits": pred_logits[b, idx, :],
                        "labels": torch.argmax(pred_logits[b, idx, :], dim=-1),
                        "output_embed": pred_queries[-1, b, idx, :],
                        "last_output": pred_queries[-1, b, idx, :],
                        "long_memory": pred_queries[-2, b, idx, :],
                        "last_appear_boxes": pred_bboxes[b, idx, :],
                        "disappear_time": torch.zeros(idx.numel(), dtype=torch.long, device=device),
                        "iou": iou,
                        "states": states,
                        **history_kwargs,
                    },
                )
            )
        return new_instances

    def forward(
        self,
        tracks: List[Track],
        decoder_out: Dict[str, Any],
        det_matched: List[Dict[str, torch.Tensor]] | None = None,
    ) -> List[Track]:
        tracks = self._update_tracks_from_main_output(tracks=tracks, out=decoder_out)

        if self.training:
            cleaned_tracks: List[Track] = []
            for track in tracks:
                if len(track) == 0:
                    cleaned_tracks.append(track)
                    continue
                # LM consumes internal DEAD tracks and one-frame training transients before propagation.
                keep = (
                    (track.ids >= 0)
                    & (track.states != int(TrackState.DEAD))
                    & (track.states != int(TrackState.NOISE))
                    & (track.states != int(TrackState.FAKE))
                )
                cleaned_tracks.append(track[keep])

            high_conf_index = self._get_high_conf_det_idx(decoder_out)
            det_instances = self._build_training_det_instances(
                out=decoder_out,
                tracks=cleaned_tracks,
                high_conf_index=high_conf_index,
                det_matched=det_matched,
            )
            next_tracks = self._select_active_tracks_train(cleaned_tracks, det_instances)
            self.frame_idx += 1
            return next_tracks

        tracks = self._drop_transient_tracks(tracks)
        high_conf_index = self._get_high_conf_det_idx(decoder_out)
        if self.det_recover_enable:
            tracks, high_conf_index = self._recover_lost_tracks_with_detection(
                tracks=tracks,
                out=decoder_out,
                high_conf_index=high_conf_index,
            )
        new_born = self._build_new_born_instances(out=decoder_out, tracks=tracks, indexes=high_conf_index)
        tracks = [t[t.states != int(TrackState.DEAD)] for t in tracks]
        self.frame_idx += 1
        return [Track.cat(t, n) for t, n in zip(tracks, new_born)]


def build(config: dict):
    sig = inspect.signature(LifeCycleManagement)
    raw_cfg = dict(config["Life_cycle_management"])
    if "Sudden_death_threshold" in raw_cfg and "sudden_death_threshold" not in raw_cfg:
        raw_cfg["sudden_death_threshold"] = raw_cfg["Sudden_death_threshold"]
    _cfg = {k: v for k, v in raw_cfg.items() if k in sig.parameters}
    return LifeCycleManagement(**_cfg)
