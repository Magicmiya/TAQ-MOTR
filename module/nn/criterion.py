# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)
# Modified from RT-DETRv2 (https://github.com/zheli-hub/RT-DETRv2)

import inspect
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Any

from .common.matcher import build as build_matcher
from .instance import TrackInstances
from .common import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from utils.utils import is_dist, dist_world_size


class Criterion(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher_cfg,
        frame_length: List[int],
        losses_weight: dict,
        aux_loss: bool,
        num_decoder_layer: int = 6,
        merge_det_track_layer: int = 0,
        hidden_dim: int = 256,
        dn_num=0,
        aux_weights: List = [],
        det_dn_aux_weights: Optional[List] = None,
        alpha=0.2,
        gamma=2.0,
        topk_disp_enable: bool = False,
        topk_disp_min_iou: float = 0.7,
        visualize=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.max_frame_length = max(frame_length)
        self.matcher = build_matcher(matcher_cfg)
        self.merge_det_track_layer = merge_det_track_layer
        self.gt_instances_list: Optional[List[List[TrackInstances]]] = None  # (clip_size, B)
        self.frame_idx = 0
        self.n_gts = []
        self.log = {}
        # losses config
        self.losses: dict[int, dict[str, torch.Tensor]] = {}
        self.losses_weight = dict(losses_weight)
        self.losses_map = {
            'loss_labels_focal': self._accumulate_focal_loss,
            'loss_labels_vfl': self._accumulate_vfl_loss,
            'loss_bboxes_L1': self._accumulate_l1_loss,
            'loss_bboxes_giou': self._accumulate_giou_loss,
        }
        self.alpha = alpha
        self.gamma = gamma
        self.topk_disp_enable = bool(topk_disp_enable)
        self.topk_disp_min_iou = float(topk_disp_min_iou)
        self.frame_weights = [1.0] * self.max_frame_length  # if you want to set different weights for different frames
        self.aux_loss = aux_loss
        self.num_aux_layer = num_decoder_layer - 1
        self.aux_weights = aux_weights  # different weights for different DETR layers
        self.det_dn_aux_weights = det_dn_aux_weights
        self.dn_loss = dn_num > 0
        self.visualize = visualize
        self._dn_group_num: int = 1
        self._label_stats: dict[int, dict[str, float]] = {}

        self.max_obj_id = 0
        if self.visualize:
            os.makedirs("./outputs/visualize_tmp/runtime_tracker/", exist_ok=True)
        self.device: None | torch.device = None
        self._loss_jobs: list[torch.cuda.Stream | None] = []

    def forward(
        self,
        decoder_out: dict[str, Any],
        tracks: List[TrackInstances],
    ):
        """

        Args:
            decoder_out (Dict[str, Dict[str, torch.Tensor]]):
            tracks (List[module.nn.instance.track_instances_V2.TrackInstances]):

        Returns:
            None:
        """

        if not self.training:
            det_matched = self._empty_det_matched(
                batch_size=len(tracks),
                device=decoder_out["pred_queries"].device,
            )
            return tracks, det_matched

        ''' ================================== for Training ================================== '''
        assert self.gt_instances_list is not None, ValueError(
            "self.gt_instances_list must be initialized by calling init_a_clip before a batch training begins"
        )
        gt_instances = self.gt_instances_list[self.frame_idx]
        n_gts = sum([len(gt) for gt in gt_instances])
        self.n_gts.append(n_gts)
        _, det_num, _ = decoder_out['meta']['query_nums']

        ''' Step 3. Temporarily update track boxes for GT matching/loss only '''
        tracks = self._refresh_tracks_for_loss(decoder_out=decoder_out, tracks=tracks)
        unmatched_gts = [t.update_with_gt(gt) for t, gt in zip(tracks, gt_instances)]

        ''' Step 4. Get Unified matching result '''
        dn_meta = decoder_out.get("meta", {}).get("dn_meta", {})
        self._dn_group_num = int(dn_meta.get("dn_num_group", 1) or 1)

        # 1) Flatten decoder outputs into unmatched/all/preset_dn groups.
        u_tree, u_outs, a_tree, a_outs, dn_tree, dn_outs = self._flatten_outputs(decoder_out)

        # 2) Run matcher in two batched calls.
        match_u = self.matcher(u_outs, unmatched_gts, re_maps=True, limit=det_num) if u_outs else []
        match_all = self.matcher(a_outs, gt_instances, re_maps=True) if a_outs else []

        # 3) Build preset DN matched indices only for GT-generated DN queries, if available.
        if dn_outs and dn_meta.get("dn_source", "gt") == "gt":
            dn_match = self._get_dn_matched_indices(dn_meta, gt_instances)
            a_tree = a_tree + dn_tree
            a_outs = a_outs + dn_outs
            match_all = match_all + [dn_match for _ in dn_tree]

        if len(u_tree) == 0 or u_tree[0] != "main_outputs":
            raise RuntimeError(f"main_outputs must be the first unmatched_gt group, got {u_tree[:1]}")
        if len(match_u) == 0:
            raise RuntimeError("main_outputs match is empty in unmatched_gt branch")

        # det_matched is the only lifecycle evidence exported by Criterion;
        # LM owns detection-query state construction and next-frame retention.
        det_matched = self._build_det_matched_main(
            out=decoder_out,
            main_match=match_u[0],
            gt_instances=gt_instances,
        )

        ''' Step 6. Get TrackInstance Matched info '''
        match_u = self._merge_track_matches(match_u, u_outs, tracks, det_num)
        tree = u_tree + a_tree
        flatten_outputs = u_outs + a_outs
        match = match_u + match_all

        ''' Step 7. Get losses '''
        if len(match) > 0:
            # Keep DN dummy_loss connected to the main loss graph for DDP, even when DN is disabled / det_dn is used.
            dummy_loss = dn_meta.get("dummy_loss", None)
            first_key = next(iter(self.losses_weight.keys()))
            if self.device is not None and self.device.type == 'cuda':
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(device=self.device))  # type: ignore
                self._record_stream_tree(flatten_outputs, stream)  # type: ignore
                self._record_stream_tree(match, stream)  # type: ignore
                self._record_stream_tree(tracks, stream)  # type: ignore

                with torch.cuda.stream(stream):  # type: ignore
                    self._record_stream_tree(decoder_out.get("qpn_main_dense_outputs", None), stream)  # type: ignore
                    self._get_losses(
                        decoder_out,
                        tree,
                        match,
                        flatten_outputs,
                        gt_instances,
                        tracks,
                        frame_idx=self.frame_idx,
                    )
                    if isinstance(dummy_loss, torch.Tensor):
                        if dummy_loss.is_cuda:
                            dummy_loss.record_stream(stream)
                        self.losses[self.frame_idx][first_key] += dummy_loss
                        self.log[self.frame_idx][first_key]["dn_dummy_loss"] = dummy_loss.detach()
                self._loss_jobs.append(stream)  # type: ignore
            else:
                self._get_losses(
                    decoder_out,
                    tree,
                    match,
                    flatten_outputs,
                    gt_instances,
                    tracks,
                    frame_idx=self.frame_idx,
                )
                if isinstance(dummy_loss, torch.Tensor):
                    self.losses[self.frame_idx][first_key] += dummy_loss
                    self.log[self.frame_idx][first_key]["dn_dummy_loss"] = dummy_loss.detach()

        ''' Step 8. Prepare outputs '''
        # Label assignment statistics (single-pass summary per frame).
        # We only keep:
        # 1) total GT count
        # 2) GT count assigned to main last-layer track queries (src >= det_num in main_outputs)
        # 3) GT count assigned to DN last-layer branch
        gt_total = int(sum(len(gt) for gt in gt_instances))

        main_track_labeled = 0
        for src_idx, tgt_idx in match_u[0]:
            if len(src_idx) == 0:
                continue
            track_mask = src_idx >= det_num
            if bool(track_mask.any()):
                main_track_labeled += int(torch.unique(tgt_idx[track_mask]).numel())

        dn_labeled = 0
        dn_match_last = None
        for group_name, group_match in zip(reversed(tree), reversed(match)):
            if group_name.startswith("det_dn_outputs_gt"):
                dn_match_last = group_match
                break
        if dn_match_last is None:
            for group_name, group_match in zip(reversed(tree), reversed(match)):
                if group_name.startswith("qpn_dn_outputs"):
                    dn_match_last = group_match
                    break
        if dn_match_last is None:
            for group_name, group_match in zip(reversed(tree), reversed(match)):
                if group_name.startswith("det_dn_outputs"):
                    dn_match_last = group_match
                    break

        if dn_match_last is not None:
            for _, tgt_idx in dn_match_last:
                if len(tgt_idx) > 0:
                    dn_labeled += int(torch.unique(tgt_idx).numel())

        self._label_stats[self.frame_idx] = {
            f"F{self.frame_idx}_gt_total": float(gt_total),
            f"F{self.frame_idx}_main_track_labeled": float(main_track_labeled),
            f"F{self.frame_idx}_dn_labeled": float(dn_labeled),
        }

        self.frame_idx += 1
        return tracks, det_matched

    @staticmethod
    def _empty_det_matched(batch_size: int, device: torch.device) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "det_query_idx": torch.zeros((0,), dtype=torch.long, device=device),
                "gt_ids": torch.zeros((0,), dtype=torch.long, device=device),
                "gt_idx": torch.zeros((0,), dtype=torch.long, device=device),
                "iou": torch.zeros((0,), dtype=torch.float32, device=device),
            }
            for _ in range(batch_size)
        ]

    def _refresh_tracks_for_loss(
        self,
        decoder_out: dict[str, Any],
        tracks: List[TrackInstances],
    ) -> List[TrackInstances]:
        """Refresh decoder-visible track predictions for GT matching without lifecycle side effects."""
        _, n_det, _ = decoder_out["meta"]["query_nums"]
        main_out = decoder_out["main_outputs"]
        for b, track in enumerate(tracks):
            if len(track) == 0:
                continue
            selected_idx = torch.arange(len(track), dtype=torch.long, device=track.device)
            pred_mask = main_out.get("query_mask", None)
            pred_mask_b = pred_mask[b, n_det:] if pred_mask is not None else None
            pred_bboxes = main_out["pred_bboxes"][b, n_det:, :]
            pred_logits = main_out["pred_logits"][b, n_det:, :]
            if selected_idx.numel() == 0:
                continue
            if pred_mask_b is not None:
                valid_idx = torch.nonzero(~pred_mask_b, as_tuple=False).squeeze(-1)
            else:
                valid_idx = torch.arange(pred_logits.shape[0], device=track.device)
            if len(valid_idx) != len(selected_idx):
                raise ValueError(
                    f"criterion track valid queries {len(valid_idx)} != selected instances {len(selected_idx)} "
                    f"(all instances={len(track)})"
                )
            mapped_idx = selected_idx.index_select(0, valid_idx)
            track.boxes[mapped_idx] = pred_bboxes[valid_idx]
            track.logits[mapped_idx] = pred_logits[valid_idx]
            track.labels[mapped_idx] = torch.argmax(pred_logits[valid_idx], dim=-1)
        return tracks

    def _build_det_matched_main(
        self,
        out: dict[str, Any],
        main_match: list[list[torch.Tensor]],
        gt_instances: list[TrackInstances],
    ) -> list[dict[str, torch.Tensor]]:
        device = out["pred_queries"].device
        _, det_num, _ = out["meta"]["query_nums"]
        pred_boxes = out["main_outputs"]["pred_bboxes"]
        det_matched = self._empty_det_matched(batch_size=len(gt_instances), device=device)
        for b, ((src_idx, tgt_idx), gt) in enumerate(zip(main_match, gt_instances)):
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            det_mask = src_idx < int(det_num)
            src_idx = src_idx[det_mask]
            tgt_idx = tgt_idx[det_mask]
            if src_idx.numel() == 0:
                continue
            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b, src_idx, :])
            gt_xyxy = box_cxcywh_to_xyxy(gt.boxes[tgt_idx])
            pair_iou = torch.diag(box_iou(pred_xyxy, gt_xyxy)[0]).to(dtype=torch.float32)
            det_matched[b] = {
                "det_query_idx": src_idx,
                "gt_ids": gt.ids[tgt_idx].to(device=device, dtype=torch.long),
                "gt_idx": tgt_idx,
                "iou": pair_iou,
            }
        return det_matched

    @staticmethod
    def _flatten_outputs(outputs: dict[str, Any]):
        """
        Split decoder outputs by matching target set.
        Returns:
            (u_tree, u_outs, a_tree, a_outs, dn_tree, dn_outs)
        """
        u_tree: list[str] = ["main_outputs"]
        u_outs: list[dict[str, torch.Tensor]] = [outputs["main_outputs"]]
        a_tree: list[str] = []
        a_outs: list[dict[str, torch.Tensor]] = []
        dn_tree: list[str] = []
        dn_outs: list[dict[str, torch.Tensor]] = []

        if "aux_outputs" in outputs:
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                u_tree.append(f"aux_outputs_{i}")
                u_outs.append(aux_out)

        if "qpn_aux_outputs" in outputs:
            qpn_aux_outputs = outputs["qpn_aux_outputs"]
            qpn_aux_unmatched = bool(outputs.get("meta", {}).get("qpn_aux_unmatched", False))
            for i, qpn_out in enumerate(qpn_aux_outputs):
                name = f"qpn_aux_outputs_{i}"
                if qpn_aux_unmatched:
                    u_tree.append(name)
                    u_outs.append(qpn_out)
                else:
                    a_tree.append(name)
                    a_outs.append(qpn_out)

        if "qpn_main_dense_outputs" in outputs:
            dense_out = dict(outputs["qpn_main_dense_outputs"])
            dense_out["det_only"] = True
            u_tree.append("main_dense")
            u_outs.append(dense_out)

        if "qpn_dn_outputs" in outputs:
            a_tree.append(f"qpn_dn_outputs")
            a_outs.append(outputs["qpn_dn_outputs"])

        if "det_dn_outputs" in outputs:
            det_dn_out = outputs["det_dn_outputs"]
            dn_meta = outputs["meta"]["dn_meta"]
            dn_source = dn_meta.get("dn_source", "gt")
            dn_list = det_dn_out if isinstance(det_dn_out, (list, tuple)) else [det_dn_out]

            for i, dn_out in enumerate(dn_list):
                is_gt_dn = dn_source == "gt"
                base_name = "det_dn_outputs_gt" if is_gt_dn else "det_dn_outputs"
                name = base_name if len(dn_list) == 1 else f"{base_name}_{i}"
                if is_gt_dn:
                    dn_tree.append(name)
                    dn_outs.append(dn_out)
                elif dn_source == "unmatched_gt":
                    u_tree.append(name)
                    u_outs.append(dn_out)
                else:
                    a_tree.append(name)
                    a_outs.append(dn_out)

        return u_tree, u_outs, a_tree, a_outs, dn_tree, dn_outs

    def _merge_track_matches(self, matches, outputs, tracks, num_det):
        bs = len(tracks)
        if bs == len(matches) and len(matches[0]) == 2 and isinstance(matches[0][0], torch.Tensor):
            for i, t in enumerate(tracks):
                track_idx = torch.arange(num_det, num_det + len(t), dtype=torch.long, device=t.device)
                t_matched_idx = t.matched_idx
                valid = t_matched_idx != -1
                matches[i][0] = torch.cat([matches[i][0], track_idx[valid]], dim=0)
                matches[i][1] = torch.cat([matches[i][1], t_matched_idx[valid]], dim=0)
            return matches
        for idx, (match, out) in enumerate(zip(matches, outputs)):
            # Some unmatched_gt branches (e.g., QPN aux) are det-only and must not receive track-index concatenation.
            # Use query length (> num_det) to gate track merge only for branches that actually include track queries.
            if out.get("det_only", False):
                continue
            if out['pred_logits'].shape[1] > num_det:
                matches[idx] = self._merge_track_matches(match, out, tracks, num_det)
        return matches

    # ================== losses  =====================
    def _get_losses(self, decoder_out, tree, matches, outputs_list, gt_instances, tracks, frame_idx: int):
        result = self._build_loss_data(tree, matches, outputs_list, gt_instances)
        if result is None:
            return
        loss_data, group_infos = result
        for loss_name in self.losses_weight:
            if loss_name in self.losses_map:
                self.losses_map[loss_name](loss_data, group_infos)
            elif loss_name == "loss_topk_disp":
                self._accumulate_topk_disp_loss(decoder_out, gt_instances)

    def _build_loss_data(self, tree, matches, outputs_list, gt_instances):
        if len(matches) == 0:
            return None
        device = outputs_list[0]['pred_logits'].device
        group_infos = []

        logits_chunks, mask_chunks, cls_group_ids = [], [], []
        target_list, pos_score_list = [], []

        bbox_preds, bbox_targets, bbox_group_ids = [], [], []
        bbox_counts = []
        bbox_group_names: list[str] = []

        for name, group_match, outputs in zip(tree, matches, outputs_list):
            group_id = len(group_infos)
            group_infos.append({"name": name, "group_id": group_id, "weight": self._get_group_weight(name)})

            logits = outputs['pred_logits']
            bs, num_queries, _ = logits.shape
            logits_flat = logits.reshape(-1, self.num_classes)
            logits_chunks.append(logits_flat)
            cls_group_ids.append(torch.full((logits_flat.shape[0],), group_id, dtype=torch.long, device=device))

            if "query_mask" in outputs:
                valid_mask = (~outputs["query_mask"]).reshape(-1, 1).to(logits.dtype)
            else:
                valid_mask = torch.ones((logits_flat.shape[0], 1), dtype=logits.dtype, device=device)
            mask_chunks.append(valid_mask)

            target_classes = torch.full((bs, num_queries), self.num_classes, dtype=torch.long, device=device)
            pos_scores = torch.zeros((bs, num_queries), dtype=logits.dtype, device=device)
            batch_indices, query_indices, target_box_chunks = [], [], []

            for b, (src_idx, tgt_idx) in enumerate(group_match):
                if len(src_idx) == 0:
                    continue
                target_classes[b, src_idx] = gt_instances[b].labels[tgt_idx]
                target_box_chunks.append(gt_instances[b].boxes[tgt_idx])
                batch_indices.append(torch.full((len(src_idx),), b, dtype=torch.long, device=device))
                query_indices.append(src_idx)

            target_list.append(target_classes)
            pos_score_list.append(pos_scores)

            if batch_indices:
                batch_idx = torch.cat(batch_indices, dim=0)
                src_idx = torch.cat(query_indices, dim=0)
                pred_boxes = outputs['pred_bboxes'][batch_idx, src_idx]
                target_boxes = torch.cat(target_box_chunks, dim=0)
                bbox_preds.append(pred_boxes)
                bbox_targets.append(target_boxes)
                bbox_group_ids.append(torch.full((len(src_idx),), group_id, dtype=torch.long, device=pred_boxes.device))
                bbox_counts.append(len(src_idx))
                bbox_group_names.append(name)
            else:
                bbox_counts.append(0)
                bbox_group_names.append(name)

        if bbox_preds:
            pred_boxes = torch.cat(bbox_preds, dim=0)
            gt_boxes = torch.cat(bbox_targets, dim=0)
            bbox_group_ids_tensor = torch.cat(bbox_group_ids, dim=0)
            boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
            targets_xyxy = box_cxcywh_to_xyxy(gt_boxes)
            ious_full, _ = box_iou(boxes_xyxy, targets_xyxy)
            iou_values = torch.diag(ious_full).detach()
            giou_values = 1 - torch.diag(generalized_box_iou(boxes_xyxy, targets_xyxy))
            iou_split = torch.split(iou_values, bbox_counts)
        else:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            bbox_group_ids_tensor = torch.zeros((0,), dtype=torch.long, device=device)
            iou_values = torch.zeros((0,), dtype=torch.float32, device=device)
            giou_values = torch.zeros((0,), dtype=torch.float32, device=device)
            iou_split = [torch.zeros((0,), dtype=torch.float32, device=device) for _ in bbox_counts]

        for pos_scores, group_match, group_iou in zip(pos_score_list, matches, iou_split):
            offset = 0
            for b, (src_idx, _) in enumerate(group_match):
                cnt = len(src_idx)
                if cnt > 0 and group_iou.numel() > 0:
                    pos_scores[b, src_idx] = group_iou[offset : offset + cnt]
                    offset += cnt

        class_data = None
        if logits_chunks:
            targets_flat, src_scores_flat = [], []
            for target_classes, pos_scores, logits in zip(target_list, pos_score_list, logits_chunks):
                target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(logits.dtype)
                targets_flat.append(target.reshape(-1, self.num_classes))
                src_scores_flat.append((pos_scores.unsqueeze(-1) * target).reshape(-1, self.num_classes))
            class_data = {
                "logits": torch.cat(logits_chunks, dim=0),
                "targets": torch.cat(targets_flat, dim=0),
                "src_scores": torch.cat(src_scores_flat, dim=0),
                "valid_mask": torch.cat(mask_chunks, dim=0),
                "group_ids": torch.cat(cls_group_ids, dim=0),
            }

        bbox_data = {
            "pred_boxes": pred_boxes,
            "gt_boxes": gt_boxes,
            "group_ids": bbox_group_ids_tensor,
            "iou_diag": iou_values,
            "giou_loss": giou_values,
        }
        return {"class_data": class_data, "bbox_data": bbox_data}, group_infos

    def _flush_loss_jobs(self):
        if not self._loss_jobs:
            return
        for stream in self._loss_jobs:
            if stream is not None:
                stream.synchronize()
        self._loss_jobs.clear()

    @staticmethod
    def _record_stream_tree(obj: Any, stream: torch.cuda.Stream) -> None:
        """
        Record CUDA tensors contained in nested Python structures onto `stream`.

        This keeps tensors alive on a non-default stream (caching allocator safety) when launching
        asynchronous loss computation.
        """
        stack = [obj]
        seen: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur is None:
                continue
            cur_id = id(cur)
            if cur_id in seen:
                continue
            seen.add(cur_id)

            if isinstance(cur, torch.Tensor):
                if cur.is_cuda:
                    cur.record_stream(stream)
                continue

            if isinstance(cur, dict):
                stack.extend(cur.values())
                continue

            if isinstance(cur, (list, tuple, set)):
                stack.extend(cur)
                continue

            tensor_attrs = getattr(cur, "_tensor_attrs", None)
            if tensor_attrs is not None:
                stack.extend(getattr(cur, name, None) for name in tensor_attrs)
                continue

    def _get_group_weight(self, group_name):
        # frame weight
        weight = self.frame_weights[self.frame_idx]
        aux_w = self._get_stage_aux_weight(group_name)
        # group weight
        if group_name.startswith("aux_outputs"):
            weight *= aux_w
        elif group_name.startswith("qpn_aux_") or group_name.startswith("qpn_dn_"):
            weight *= aux_w
        elif group_name.startswith("det_dn_outputs_gt"):
            weight *= aux_w
            weight /= float(self._dn_group_num if self._dn_group_num > 0 else 1)
        elif group_name.startswith("det_dn_outputs"):
            weight *= aux_w
        return weight

    @staticmethod
    def _get_layer_weight(group_name: str, weights: Optional[List], fallback_last: bool = False) -> float:
        if not weights:
            return 1.0
        try:
            idx = int(group_name.rsplit("_", 1)[-1])
        except Exception:
            idx = len(weights) - 1 if fallback_last else 0
        if 0 <= idx < len(weights):
            return float(weights[idx])
        return 1.0

    def _get_stage_aux_weight(self, group_name: str) -> float:
        # det_dn can now use a dedicated stage-aware weight list without affecting main/qpn aux branches.
        if group_name.startswith("det_dn_outputs"):
            det_dn_weights = self.det_dn_aux_weights if self.det_dn_aux_weights is not None else self.aux_weights
            return self._get_layer_weight(group_name, det_dn_weights, fallback_last=True)
        return self._get_layer_weight(group_name, self.aux_weights, fallback_last=False)

    @staticmethod
    def _get_dn_matched_indices(meta, gts):
        """get_dn_matched_indices"""
        dn_positive_idx, dn_num_group = meta["dn_positive_idx"], meta["dn_num_group"]
        num_gts = [len(gt.labels) for gt in gts]
        device = gts[0].labels.device

        dn_match_indices = []
        for b, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[b]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[b], gt_idx))
            else:
                dn_match_indices.append(
                    (torch.zeros(0, dtype=torch.int64, device=device), torch.zeros(0, dtype=torch.int64, device=device))
                )
        return dn_match_indices

    def _get_weight(self, loss_name):
        if "bboxes_L1" in loss_name:
            return self.losses_weight["loss_bboxes_L1"]
        elif "bboxes_giou" in loss_name:
            return self.losses_weight["loss_bboxes_giou"]
        elif "labels_focal" in loss_name:
            return self.losses_weight["loss_labels_focal"]
        elif "labels_vfl" in loss_name:
            return self.losses_weight["loss_labels_vfl"]
        else:
            return 1.0

    # ================== losses kernel =====================
    def _accumulate_focal_loss(self, loss_data, group_infos):
        class_data = loss_data["class_data"]
        if class_data is None:
            return
        logits = class_data["logits"]
        if logits.numel() == 0:
            return
        targets = class_data["targets"]
        valid_mask = class_data["valid_mask"].squeeze(1)
        losses = torchvision.ops.sigmoid_focal_loss(logits, targets, self.alpha, self.gamma, reduction='none')
        losses = losses.mean(1) * valid_mask
        group_ids = class_data["group_ids"]
        group_summed = torch.zeros(len(group_infos), dtype=losses.dtype, device=losses.device)
        group_summed.scatter_add_(0, group_ids, losses)

        for info in group_infos:
            val = group_summed[info['group_id']] * info['weight'] * self.losses_weight['loss_labels_focal']
            self.losses[self.frame_idx]['loss_labels_focal'] += val
            self.log[self.frame_idx]['loss_labels_focal'][info['name']] = val.detach()

    def _accumulate_vfl_loss(self, loss_data: dict[str, dict[str, torch.Tensor]], group_infos: List[dict[str, Any]]):
        class_data = loss_data["class_data"]
        if class_data is None:
            return
        logits = class_data["logits"]
        if logits.numel() == 0:
            return
        targets = class_data["targets"]
        src_scores = class_data["src_scores"]
        valid_mask = class_data["valid_mask"].squeeze(1)
        group_ids = class_data["group_ids"]

        pred_score = torch.sigmoid(logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - targets) + src_scores
        losses = F.binary_cross_entropy_with_logits(logits, src_scores, weight=weight, reduction='none')
        losses = losses.mean(1) * valid_mask

        group_summed = torch.zeros(len(group_infos), dtype=losses.dtype, device=losses.device)
        group_summed.scatter_add_(0, group_ids, losses)

        for info in group_infos:
            val = group_summed[info['group_id']] * info['weight'] * self.losses_weight['loss_labels_vfl']
            self.losses[self.frame_idx]['loss_labels_vfl'] += val
            self.log[self.frame_idx]['loss_labels_vfl'][info['name']] = val.detach()

    def _accumulate_l1_loss(self, loss_data: dict[str, dict[str, torch.Tensor]], group_infos: List[dict[str, Any]]):
        bbox_data = loss_data["bbox_data"]
        pred_boxes = bbox_data["pred_boxes"]
        if pred_boxes.numel() == 0:
            return
        target_boxes = bbox_data["gt_boxes"]
        values = F.l1_loss(pred_boxes, target_boxes, reduction='none').sum(dim=1)
        group_ids = bbox_data["group_ids"]
        group_summed = torch.zeros(len(group_infos), dtype=values.dtype, device=values.device)
        group_summed.scatter_add_(0, group_ids, values)

        for info in group_infos:
            val = group_summed[info['group_id']] * info['weight'] * self.losses_weight['loss_bboxes_L1']
            self.losses[self.frame_idx]['loss_bboxes_L1'] += val
            self.log[self.frame_idx]['loss_bboxes_L1'][info['name']] = val.detach()

    def _accumulate_giou_loss(self, loss_data: dict[str, dict[str, torch.Tensor]], group_infos: List[dict[str, Any]]):
        bbox_data = loss_data["bbox_data"]
        giou_loss = bbox_data["giou_loss"]
        if giou_loss.numel() == 0:
            return
        group_ids = bbox_data["group_ids"]
        group_summed = torch.zeros(len(group_infos), dtype=giou_loss.dtype, device=giou_loss.device)
        group_summed.scatter_add_(0, group_ids, giou_loss)

        for info in group_infos:
            val = group_summed[info['group_id']] * info['weight'] * self.losses_weight['loss_bboxes_giou']
            self.losses[self.frame_idx]['loss_bboxes_giou'] += val
            self.log[self.frame_idx]['loss_bboxes_giou'][info['name']] = val.detach()

    def _accumulate_topk_disp_loss(self, decoder_out: dict[str, Any], gt_instances: list[TrackInstances]):
        if not self.topk_disp_enable or "loss_topk_disp" not in self.losses_weight:
            return

        log_key = "main_dense"
        dense_out = decoder_out.get("qpn_main_dense_outputs", None)
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        if not isinstance(dense_out, dict):
            self.log[self.frame_idx]["loss_topk_disp"][log_key] = zero.detach()
            return

        pred_logits = dense_out["pred_logits"]
        pred_bboxes = dense_out["pred_bboxes"]
        query_mask = dense_out.get("query_mask", None)
        score_logits = pred_logits.squeeze(-1) if pred_logits.shape[-1] == 1 else pred_logits.max(dim=-1).values
        pred_scores = torch.sigmoid(score_logits)
        # Keep a graph-connected zero so every DDP rank traverses the same parameter graph
        # even when this batch contributes no valid dispersion samples.
        graph_zero = pred_scores.sum() * 0.0

        batch_losses: list[torch.Tensor] = []
        for b, gt in enumerate(gt_instances):
            if len(gt) == 0:
                continue

            if query_mask is None:
                valid_mask = torch.ones((pred_logits.shape[1],), dtype=torch.bool, device=pred_logits.device)
            else:
                valid_mask = ~query_mask[b]
            if not bool(valid_mask.any()):
                continue

            boxes = pred_bboxes[b, valid_mask].detach()
            scores = pred_scores[b, valid_mask]
            gt_boxes = gt.boxes
            if boxes.numel() == 0 or gt_boxes.numel() == 0:
                continue

            ious, _ = box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt_boxes))
            max_iou, assigned_gt = ious.max(dim=1)
            keep = max_iou >= self.topk_disp_min_iou
            if not bool(keep.any()):
                continue

            assigned_gt = assigned_gt[keep]
            scores = scores[keep]
            unique_gt, inverse = torch.unique(assigned_gt, return_inverse=True)
            effective_gt = int(unique_gt.numel())
            if effective_gt <= 1:
                continue

            score_mass = torch.zeros((effective_gt,), dtype=scores.dtype, device=scores.device)
            score_mass.scatter_add_(0, inverse, scores)
            total_mass = score_mass.sum()
            if not bool(total_mass > 0):
                continue

            dist = score_mass / total_mass.clamp_min(1e-6)
            target = torch.full_like(dist, 1.0 / effective_gt)
            batch_losses.append(F.mse_loss(dist, target, reduction="mean"))

        if batch_losses:
            raw_loss = graph_zero + torch.stack(batch_losses).mean()
        else:
            raw_loss = graph_zero

        val = raw_loss * self.frame_weights[self.frame_idx] * self.losses_weight["loss_topk_disp"]
        self.losses[self.frame_idx]["loss_topk_disp"] += val
        self.log[self.frame_idx]["loss_topk_disp"][log_key] = val.detach()

    # ================== API  =====================
    def init_a_clip(self, batch: Dict, device: torch.device):
        """Init this function for a specific clip and build gt_instances_list
        Args:
            batch: a batch data.
            device:
        Returns:
        """
        self.device = device
        self._flush_loss_jobs()  # assert last epoch finish
        self.n_gts = []
        infos = batch["infos"]
        clip_size = len(infos)
        batch_size = len(infos[0])
        self.gt_instances_list = []
        # We use "matched_idx" to store the idx of the id in gt
        for c in range(clip_size):
            self.gt_instances_list.append(
                TrackInstances.init_tracks(
                    batch,
                    self.hidden_dim,
                    self.num_classes,
                    device,
                    kwargs=[
                        {
                            "ids": infos[c][b]["ids"],
                            "labels": infos[c][b]["labels"],
                            "boxes": infos[c][b]["bbox"],
                            "matched_idx": torch.arange(infos[c][b]["ids"].shape[0], dtype=torch.int32, device=device),
                        }
                        for b in range(batch_size)
                    ],
                )  # type: ignore
            )

        if self.training:
            self.frame_idx = 0
            self._label_stats = {}
            '''Initialize losses directly on target device'''
            self.losses = {
                c: {
                    str(loss_name): torch.zeros((), dtype=torch.float32, device=device)
                    for loss_name in self.losses_weight
                }
                for c in range(clip_size)
            }
            self.log = {c: {loss_name: {} for loss_name in self.losses_weight} for c in range(clip_size)}
        elif self.frame_idx == 0:
            # Keep ID assignment stable across frames; reset only at the start of a video/clip.
            self.max_obj_id = 0
        return

    def get_normed_loss(self) -> Tuple[float, Dict]:
        """Calculate mean losses by number of ground truths"""
        self._flush_loss_jobs()
        n_gts = torch.as_tensor(self.n_gts, dtype=torch.float, device=self.device)
        if is_dist():
            torch.distributed.all_reduce(n_gts)  # type: ignore
        total_n_gts = torch.clamp(sum(n_gts) / dist_world_size(), min=1).item()  # type: ignore
        # n_gts = torch.clamp(n_gts / distributed_world_size(), min=1).tolist()

        # Normalize losses
        loss = sum([sum([self.losses[f][k] / total_n_gts for k in self.losses[f]]) for f in self.losses])
        # Normalize logs and add weighted losses for tensorboard verification
        for f, f_val in self.log.items():
            for loss_name, loss_val in f_val.items():
                for sub_name, sub_val in loss_val.items():
                    self.log[f][loss_name][sub_name] = sub_val / total_n_gts
        log = deepcopy(self.log)
        # Extra TensorBoard-only scalars
        tb_scalars: dict[str, float] = {}
        for _, stats in self._label_stats.items():
            tb_scalars.update(stats)
        log["__tb_scalars__"] = {"label_stats": tb_scalars}  # type: ignore

        return loss, log


def build(config: dict):
    sig = inspect.signature(Criterion)
    _cfg = {k: v for k, v in config['Criterion'].items() if k in sig.parameters}
    return Criterion(**_cfg)
