# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modules to compute the matching cost and solve the corresponding LSAP.
# ------------------------------------------------------------------------
# Copyright(c) 2023 lyuwenyu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)
# Modified from RT-DETRv2 (https://github.com/zheli-hub/RT-DETRv2)
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import inspect
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from ..instance import TrackInstances


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets, re_maps: bool = False, limit: int | None = None):
        """Performs the matching

        Params:
            outputs: A dict or a list of dicts. Each dict must contain:
                 "pred_logits": Tensor [B, num_queries, num_classes]
                 "pred_bboxes"/"pred_boxes": Tensor [B, num_queries, 4]
                 "query_mask" (optional): Bool Tensor [B, num_queries] marking invalid queries.
            targets: List (len = batch_size). Each element is TrackInstances or dict with
                 "labels" and "boxes".
            re_maps: Optional list (len = batch_size). Map from target indices in `targets`
                 to indices in the original GT set (e.g. unmatched->full gt).

        Returns:
            If inputs is a dict -> List (len = batch_size) of [index_i, index_j].
            If inputs is a list -> List (len = num_outputs) where each element is the
            same structure as above (per output group).
        """
        with torch.no_grad():
            single_output = isinstance(outputs, dict)
            if single_output:
                outputs_list = [outputs]
            else:
                assert isinstance(outputs, (list, tuple)), "outputs must be a dict or a list/tuple of dicts."
                outputs_list = list(outputs)
                assert len(outputs_list) > 0, "outputs list should not be empty."

            batch_size = outputs_list[0]["pred_logits"].shape[0]
            assert len(targets) == batch_size, "targets length must match batch size."
            device = outputs_list[0]["pred_logits"].device
            num_groups = len(outputs_list)

            # Concat target labels and boxes
            tgt_labels_chunks, tgt_boxes_chunks, tgt_sizes, tgt_idx_map = [], [], [], []
            for gt_per_img in targets:
                if isinstance(targets[0], TrackInstances):
                    tgt_labels_chunks.append(gt_per_img.labels)
                    tgt_boxes_chunks.append(gt_per_img.boxes)
                    tgt_sizes.append(len(gt_per_img))
                    tgt_idx_map.append(gt_per_img.matched_idx if re_maps else None)
                else:
                    tgt_labels_chunks.append(gt_per_img["labels"])
                    tgt_boxes_chunks.append(gt_per_img["boxes"])
                    tgt_sizes.append(len(gt_per_img["boxes"]))
                    tgt_idx_map.append(gt_per_img["matched_idx"] if re_maps else None)
            total_tgt = sum(tgt_sizes)
            if total_tgt == 0:
                empty = torch.zeros(0, dtype=torch.long, device=device)
                if single_output:
                    return [[empty, empty] for _ in range(batch_size)]
                else:
                    return [[[empty.clone(), empty.clone()] for _ in range(batch_size)] for _ in range(num_groups)]
            tgt_labels = torch.cat(tgt_labels_chunks, dim=0)
            tgt_boxes = torch.cat(tgt_boxes_chunks, dim=0)

            # Also concat outputs labels and boxes
            logits_list, boxes_list, masks, group_sizes = [], [], [], []
            for group in outputs_list:
                logits = group["pred_logits"]
                boxes = group["pred_bboxes"]
                assert logits.shape[:2] == boxes.shape[:2], "pred_logits and pred_boxes dimensions must match."
                assert logits.shape[0] == batch_size, "Batch size mismatch between outputs."
                if limit is not None:
                    logits_list.append(logits[:, :limit, :])
                    boxes_list.append(boxes[:, :limit, :])
                    group_sizes.append(limit)
                    _mask = group.get("query_mask")
                    masks.append(_mask[:, :limit] if _mask is not None else None)
                else:
                    logits_list.append(logits)
                    boxes_list.append(boxes)
                    group_sizes.append(logits.shape[1])
                    masks.append(group.get("query_mask"))

            total_queries = sum(group_sizes)
            if total_queries == 0:
                empty = torch.zeros(0, dtype=torch.long, device=device)
                if single_output:
                    return [[empty, empty] for _ in range(batch_size)]
                return [[[empty.clone(), empty.clone()] for _ in range(batch_size)] for _ in range(num_groups)]
            concat_logits = torch.cat(logits_list, dim=1)
            concat_boxes = torch.cat(boxes_list, dim=1)

            if self.use_focal_loss:
                out_prob = concat_logits.flatten(0, 1).sigmoid()
            else:
                out_prob = concat_logits.flatten(0, 1).softmax(-1)  # [B * total_queries, num_classes]
            out_bbox = concat_boxes.flatten(0, 1)  # [B * total_queries, 4]
            # Compute classification cost
            if self.use_focal_loss:
                neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_labels] - neg_cost_class[:, tgt_labels]
            else:
                cost_class = -out_prob[:, tgt_labels]

            # Regression costs
            cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_boxes))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(batch_size, total_queries, -1)

            # Mask out invalid queries (tracks, paddings, etc.)
            if any(mask is not None for mask in masks):
                prepared_masks = []
                for mask, size in zip(masks, group_sizes):
                    if mask is None:
                        prepared_masks.append(torch.zeros(batch_size, size, dtype=torch.bool, device=device))
                    else:
                        prepared_masks.append(mask)
                query_mask = torch.cat(prepared_masks, dim=1)
                C = C.masked_fill(query_mask.unsqueeze(-1), torch.finfo(C.dtype).max)

            # Build offsets to slice query dimension per group
            query_offsets = [0]
            for size in group_sizes:
                query_offsets.append(query_offsets[-1] + size)

            # Prepare result containers
            group_matches: list[list[list[torch.Tensor]]] = [[] for _ in range(num_groups)]

            tgt_offset = 0
            for b, tgt_size in enumerate(tgt_sizes):
                tgt_slice = slice(tgt_offset, tgt_offset + tgt_size)
                tgt_base = tgt_offset
                tgt_offset += tgt_size

                for g in range(num_groups):
                    q_start, q_end = query_offsets[g], query_offsets[g + 1]
                    pred_slice = slice(q_start, q_end)

                    # get matched index
                    if tgt_size == 0 or q_end - q_start == 0:
                        pred_idx = torch.zeros(0, dtype=torch.long, device=device)
                        tgt_idx_local = torch.zeros(0, dtype=torch.long, device=device)
                    else:
                        cost = C[b, pred_slice, tgt_slice]
                        if cost.numel() == 0:
                            pred_idx = torch.zeros(0, dtype=torch.long, device=device)
                            tgt_idx_local = torch.zeros(0, dtype=torch.long, device=device)
                        else:
                            valid_rows = None
                            if any(mask is not None for mask in masks):
                                group_mask = prepared_masks[g][b]
                                valid_rows = torch.nonzero(~group_mask, as_tuple=False).squeeze(-1)
                            if valid_rows is not None and valid_rows.numel() == 0:
                                pred_idx = torch.zeros(0, dtype=torch.long, device=device)
                                tgt_idx_local = torch.zeros(0, dtype=torch.long, device=device)
                                if tgt_idx_map is not None:
                                    assert len(tgt_idx_map) == len(targets)
                                    mapped = tgt_idx_local
                                else:
                                    mapped = tgt_idx_local
                                group_matches[g].append([pred_idx, mapped])
                                continue
                            if valid_rows is not None:
                                cost = cost.index_select(0, valid_rows)
                            cost_cpu = cost.to(device="cpu", dtype=torch.float32)
                            row_ind, col_ind = linear_sum_assignment(cost_cpu.numpy())
                            pred_idx = torch.as_tensor(row_ind, dtype=torch.long, device=device)
                            tgt_idx_local = torch.as_tensor(col_ind, dtype=torch.long, device=device)
                            if valid_rows is not None and pred_idx.numel() > 0:
                                pred_idx = valid_rows.index_select(0, pred_idx)

                    if tgt_idx_map is not None:
                        assert len(tgt_idx_map) == len(targets)
                        mapped = tgt_idx_map[b][tgt_idx_local] if tgt_idx_local.numel() > 0 else tgt_idx_local
                    else:
                        mapped = tgt_idx_local

                    group_matches[g].append([pred_idx, mapped])

            result = group_matches[0] if single_output else group_matches
            return result

def build(config: dict):
    sig = inspect.signature(HungarianMatcher)
    _cfg = {k: v for k, v in config.items() if k in sig.parameters}
    return HungarianMatcher(**_cfg)
