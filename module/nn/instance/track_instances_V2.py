# Copyright (c) Ruopeng Gao. All Rights Reserved.
# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)
import torch
import numpy as np
from typing import MutableMapping, List, Optional, Any
from collections import defaultdict
from .base_instances_V2 import BaseInstances
from ..common import box_cxcywh_to_xyxy, box_iou, logits_to_scores


class TrackInstances(BaseInstances):
    """
    Tracked Instances.
    """

    def __init__(
            self, frame_height: float, frame_width: float, hidden_dim: int = 256, num_classes: int = 1,
            frame=0, device=torch.device("cpu"), **kwargs
    ):
        super().__init__(frame_height, frame_width, hidden_dim, num_classes, frame, device, **kwargs)
        self._gt_id2idx: Optional[torch.Tensor] = None

    def __repr__(self):
        return f"frame{self.frame:0>{4}}-{self._len}-{self.ids.tolist()}"

    def push_history_(
        self,
        indices: torch.Tensor,
        frame_idx: int,
        boxes: torch.Tensor,
        output_embed: Optional[torch.Tensor] = None,
    ) -> None:
        """Push detached observations into the per-track FIFO history."""
        if self._len == 0 or indices.numel() == 0:
            return

        indices = indices.to(device=self.device, dtype=torch.long)
        boxes = boxes.detach().to(device=self.device, dtype=torch.float32)
        slots = self.hist_ptr.index_select(0, indices)
        frame_tensor = torch.full((indices.numel(),), int(frame_idx), dtype=torch.long, device=self.device)

        self.hist_boxes[indices, slots] = boxes
        self.hist_frame_idx[indices, slots] = frame_tensor

        if output_embed is not None:
            self.hist_output_embed[indices, slots] = output_embed.detach().to(device=self.device, dtype=torch.float32)

        self.hist_ptr[indices] = (slots + 1) % self.history_len
        self.hist_count[indices] = torch.clamp(
            self.hist_count.index_select(0, indices) + 1,
            max=self.history_len,
        )

    def get_history(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return valid history ordered from oldest to newest for one track."""
        if not (0 <= index < self._len):
            raise IndexError(f"History index {index} out of range for length {self._len}")

        count = int(self.hist_count[index].item())
        if count <= 0:
            empty_frames = torch.zeros((0,), dtype=torch.long, device=self.device)
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
            return empty_frames, empty_boxes

        ptr = int(self.hist_ptr[index].item())
        order = (torch.arange(count, device=self.device, dtype=torch.long) + (ptr - count)) % self.history_len
        frames = self.hist_frame_idx[index].index_select(0, order)
        boxes = self.hist_boxes[index].index_select(0, order)
        return frames, boxes

    def _get_gt_id_map(self) -> Optional[torch.Tensor]:
        """
        Build (or reuse) a dense id->index map for GT instances.
        """
        if self._gt_id2idx is not None:
            return self._gt_id2idx
        if self._len == 0:
            self._gt_id2idx = torch.zeros((0,), dtype=torch.long, device=self.device)
            return self._gt_id2idx

        valid_mask = self.ids >= 0
        if not valid_mask.any():
            self._gt_id2idx = torch.zeros((0,), dtype=torch.long, device=self.device)
            return self._gt_id2idx

        valid_ids = self.ids[valid_mask]
        max_id = torch.max(valid_ids).item()
        map_tensor = torch.full((max_id + 1,), -1, dtype=torch.long, device=self.device) # type: ignore
        map_tensor[valid_ids] = torch.arange(len(self.ids), device=self.device, dtype=torch.long)[valid_mask]
        self._gt_id2idx = map_tensor
        return self._gt_id2idx

    def update_with_gt(self, gt: "TrackInstances"):
        """
        Update matched_idx/iou using GT data and return unmatched GT instances.
        """
        gt_len = len(gt)
        if self._len > 0:
            self.matched_idx.fill_(-1)
            self.iou.fill_(-1)
        if self._len == 0 or gt_len == 0:
            return gt

        unmatched_mask = torch.ones(gt_len, dtype=torch.bool, device=gt.device)
        id_map = gt._get_gt_id_map()
        if id_map is None or id_map.numel() == 0:
            return gt

        id_tensor = self.ids
        valid_ids = (id_tensor >= 0) & (id_tensor < id_map.shape[0])
        mapped_idx = torch.full((self._len,), -1, dtype=torch.long, device=self.device)
        if valid_ids.any():
            mapped_idx[valid_ids] = id_map[id_tensor[valid_ids]]

        matched_mask = mapped_idx >= 0
        if matched_mask.any():
            gt_idx = mapped_idx[matched_mask]
            self.matched_idx[matched_mask] = gt_idx

            matched_boxes = self.boxes[matched_mask]
            gt_boxes = gt.boxes[gt_idx]
            ious = box_iou(
                box_cxcywh_to_xyxy(matched_boxes),
                box_cxcywh_to_xyxy(gt_boxes))[0]
            self.iou[matched_mask] = torch.diag(ious)
            unmatched_mask[gt_idx] = False

        return gt[unmatched_mask]

    def update_with_out(self, out, b, frame, track_conf: float, miss_tolerance: int):
        self.frame = frame
        self.last_appear_boxes = self.boxes.clone()
        if self._len == 0:
            return self
        # track part starts after detection queries
        _, n_det, _ = out['meta']['query_nums']
        pred_mask = out['main_outputs']['query_mask'][b, n_det:] if 'query_mask' in out['main_outputs'] else None
        pred_bboxes = out['main_outputs']['pred_bboxes'][b, n_det:, :]
        pred_logits = out['main_outputs']['pred_logits'][b, n_det:, :]
        pred_queries = out['pred_queries'][:, b, n_det:, :]

        # select valid indices (non-masked), fallback to sequential slice
        if pred_mask is not None:
            valid_idx = torch.nonzero(~pred_mask, as_tuple=False).squeeze(-1)
        else:
            valid_idx = torch.arange(pred_logits.shape[0], device=self.device)
        if len(valid_idx) != self._len:
            raise ValueError(f"track valid queries {len(valid_idx)} != instances {self._len}")

        self.boxes = pred_bboxes[valid_idx]
        self.logits = pred_logits[valid_idx]
        self.output_embed = pred_queries[-1, valid_idx, :]

        scores = self.scores
        conf_mask = scores >= track_conf
        new_labels = torch.argmax(self.logits, dim=-1)
        label_changed = self.labels != new_labels
        self.labels.copy_(new_labels)

        survive_mask = conf_mask & (~label_changed)
        self.disappear_time[~survive_mask] += 1
        self.disappear_time[survive_mask] = 0
        self.last_appear_boxes[survive_mask] = self.boxes[survive_mask]
        self.ids[self.disappear_time >= miss_tolerance] = -1

        return self

    def result(self, only_active: bool = False, min_score: Optional[float] = None, min_area: Optional[float] = None,
               frame_width: Optional[float] = None, frame_height: Optional[float] = None, ) -> np.ndarray:
        """Export current tracks in MOTChallenge txt format order.

        Args:
            frame_height (int):
            frame_width (int):
            only_active: If True, only export active tracks (not disappeared in current frame).
            min_score: Minimum score threshold for exporting.
            min_area: Minimum bbox area threshold in pixel space (w*h) after scaling.
        """
        if self._len == 0:
            return np.empty((0, 9), dtype=np.float32)

        states = self.states if self.states is not None else torch.full((self._len,), 4, dtype=torch.long, device=self.device)
        # Visible states: NEW/RELIABLE/RECOVER/CONFUSED/LOST.
        valid_mask = states < 5
        if only_active:
            # Active states: NEW/RELIABLE/RECOVER/CONFUSED.
            valid_mask = valid_mask & (states <= 3)
        if not valid_mask.any():
            return np.empty((0, 9), dtype=np.float32)

        boxes = self.boxes[valid_mask]
        ids = self.ids[valid_mask]
        scores = self.scores[valid_mask]
        labels = self.labels[valid_mask]

        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        fw = float(self.frame_width) if frame_width is None else float(frame_width)
        fh = float(self.frame_height) if frame_height is None else float(frame_height)
        scale = torch.as_tensor(
            [fw, fh, fw, fh],
            dtype=boxes_xyxy.dtype,
            device=self.device
        )
        boxes_xyxy = boxes_xyxy * scale
        boxes_xywh = torch.cat([boxes_xyxy[:, :2], boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]], dim=1)

        if min_score is not None:
            keep = scores >= float(min_score)
            if not keep.any():
                return np.empty((0, 9), dtype=np.float32)
            boxes_xywh = boxes_xywh[keep]
            ids = ids[keep]
            scores = scores[keep]
            labels = labels[keep]

        if min_area is not None and float(min_area) > 0:
            area = boxes_xywh[:, 2] * boxes_xywh[:, 3]
            keep = area >= float(min_area)
            if not keep.any():
                return np.empty((0, 9), dtype=np.float32)
            boxes_xywh = boxes_xywh[keep]
            ids = ids[keep]
            scores = scores[keep]
            labels = labels[keep]

        frame_col = torch.full((len(ids), 1), self.frame, dtype=torch.long, device=self.device)
        visibility_col = torch.ones((len(ids), 1), dtype=torch.long, device=self.device)

        result_cols = [
            frame_col,
            ids.unsqueeze(1),
            boxes_xywh,
            scores.unsqueeze(1),
            labels.unsqueeze(1),
            visibility_col
        ]
        result = torch.cat(result_cols, dim=1).detach().to('cpu').numpy().astype(np.float32, copy=False)
        return result
