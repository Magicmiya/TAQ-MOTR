# Modified from RT-DETRv2 (https://github.com/zheli-hub/RT-DETRv2)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict

from utils.visualizer import TensorHook
from ..common import MLP, inverse_sigmoid, bias_init_with_prob, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from ..instance import TrackInstances


class QueryGenerator(nn.Module):
    """
    Unified Query Generator for DETR-based Multi-Object Tracking

    Integrates three types of query generation:
    1. Denoising queries (from ground truth)
    2. Detection queries (learnable embeddings or generative)
    3. Tracker queries (from previous frames)
    """

    def __init__(
        self,
        # Basic parameters
        hidden_dim=256,
        num_queries=300,
        num_classes=80,
        # Query generation modes
        mode="learnable",  # 'learnable', 'generative'
        query_select_method="default",
        # Denoising parameters
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        use_det_dn: bool = False,
        # Generative parameters
        feat_strides=[8, 16, 32],
        eval_spatial_size=None,
        eps=1e-5,
    ):
        super(QueryGenerator, self).__init__()

        # Basic configuration
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.mode = mode
        self.query_select_method = query_select_method
        self.eps = eps

        # Denoising configuration
        # When using det_dn bypass, force GT DN count to 0 (still call _get_denoising_queries for dummy_loss).
        self.num_denoising = 0 if use_det_dn else num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.use_det_dn = use_det_dn

        # Generative configuration
        self.feat_strides = feat_strides
        self.eval_spatial_size = eval_spatial_size

        """ Initialize components based on mode """

        # Denoising class embedding (always needed)
        self.denoising_class_embed = nn.Embedding(self.num_classes + 1, self.hidden_dim, padding_idx=self.num_classes)

        # Detection query components
        if self.mode == "learnable":
            # Learnable embeddings
            self.det_content = nn.Embedding(self.num_queries, self.hidden_dim)
            self.det_anchor = nn.Embedding(self.num_queries, 4)
        elif self.mode == "generative":
            self.enc_output = nn.Sequential(
                OrderedDict(
                    [
                        ("proj", nn.Linear(self.hidden_dim, self.hidden_dim)),
                        ("norm", nn.LayerNorm(self.hidden_dim)),
                    ]
                )
            )
            head_dim = 1 if self.query_select_method == "agnostic" else self.num_classes
            self.enc_score_head = nn.Linear(self.hidden_dim, head_dim)
            self.enc_bbox_head = MLP(self.hidden_dim, self.hidden_dim, 4, 3)

            # Pre-computed anchors for evaluation
            if self.eval_spatial_size:
                anchors, valid_mask = self._generate_anchors()
                self.register_buffer("anchors", anchors)
                self.register_buffer("valid_mask", valid_mask)
        else:
            raise NotImplementedError("mode {} not supported".format(self.mode))
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        bias = bias_init_with_prob(0.01)

        # Encoder output projection
        if hasattr(self, "enc_output"):
            nn.init.xavier_uniform_(self.enc_output[0].weight)

        # Denoising class embedding
        if hasattr(self, "denoising_class_embed"):
            nn.init.normal_(self.denoising_class_embed.weight[:-1])

        # Detection query components
        if self.mode == "learnable":
            nn.init.xavier_uniform_(self.det_content.weight)
            nn.init.normal_(self.det_anchor.weight)
        elif self.mode == "generative":
            if hasattr(self, "enc_score_head"):
                nn.init.constant_(self.enc_score_head.bias, bias)
            if hasattr(self, "enc_bbox_head"):
                nn.init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
                nn.init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        else:
            raise NotImplementedError("mode {} not supported".format(self.mode))

    def stopDN(self) -> bool:
        """
        Stop DN branch generation at runtime.

        Returns:
            bool: True if any DN-related flag/state was changed in this call.
        """
        changed = False
        if hasattr(self, "use_det_dn") and bool(self.use_det_dn):
            self.use_det_dn = False
            changed = True
        if hasattr(self, "use_dn") and bool(getattr(self, "use_dn")):
            self.use_dn = False
            changed = True
        if int(getattr(self, "num_denoising", 0)) != 0:
            self.num_denoising = 0
            changed = True
        self._dn_stopped = True
        return changed

    def _get_trackers(self, trackers: List[TrackInstances]):
        """
        Get tracker query inputs for the decoder.

        Args:
            trackers (List[TrackInstances]): List of track instances for each batch item

        Returns:
            tracker_anchors (torch.Tensor): Tracker query anchors in logits space，global coords
            tracker_contents (torch.Tensor): Tracker query content embeddings
            tracker_mask (torch.Tensor): Tracker query padding mask (True for padding queries)
        """
        bs = len(trackers)
        device = self.denoising_class_embed.weight.device
        max_tracks = max([len(t) for t in trackers])
        tracker_anchors = torch.zeros((bs, max_tracks, 4), dtype=torch.float, device=device)
        tracker_contents = torch.zeros((bs, max_tracks, self.hidden_dim), dtype=torch.float, device=device)
        tracker_mask = torch.zeros((bs, max_tracks), dtype=torch.bool, device=device)
        for i in range(bs):
            tracker_len = len(trackers[i])
            tracker_anchors[i, :tracker_len, :] = trackers[i].ref_pts
            tracker_contents[i, :tracker_len, :] = trackers[i].query_embed
            tracker_mask[i, tracker_len:] = True  # padding fake track query, mark as True
        return tracker_anchors.detach(), tracker_contents, tracker_mask

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, device=torch.device("cpu")):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])
            spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=device)
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=torch.float32, device=device)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)
        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _select_topK(self, contents: torch.Tensor, logits: torch.Tensor, anchors: torch.Tensor, topk: int):
        if self.query_select_method == "default":
            _, topK_ind = torch.topk(logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topK_ind = torch.topk(logits.flatten(1), topk, dim=-1)
            topK_ind = topK_ind // self.num_classes
        elif self.query_select_method == "agnostic":
            _, topK_ind = torch.topk(logits.squeeze(-1), topk, dim=-1)
        else:
            _, topK_ind = torch.topk(logits.max(-1).values, topk, dim=-1)

        topK_anchors = anchors.gather(dim=1, index=topK_ind.unsqueeze(-1).repeat(1, 1, anchors.shape[-1]))
        topK_logits = logits.gather(dim=1, index=topK_ind.unsqueeze(-1).repeat(1, 1, logits.shape[-1]))
        topK_contents = contents.gather(dim=1, index=topK_ind.unsqueeze(-1).repeat(1, 1, contents.shape[-1]))

        return topK_contents, topK_logits, topK_anchors

    def _get_denoising_queries(self, targets: List[dict], valid_ratios: Optional[torch.Tensor] = None):
        """
        Get detection denoising query form gt for the decoder.

        Args:
            targets (list[dict]): GT targets with boxes in valid image region coordinates [0,1]
            valid_ratios (torch.Tensor): [b, n_levels, 2] Valid ratios to convert coordinates to global image space

        Returns:
            dn_anchors (torch.Tensor): Denoising query anchors in logits space (inverse_sigmoid format)
                                      Converted from GT boxes [0,1] -> global coords -> logits space
            dn_contents (torch.Tensor): Denoising query content embeddings
            dn_mask (torch.Tensor): Denoising query padding mask (True for padding queries)
            dn_meta (Dict): Metadata containing denoising group information and positive indices
        """
        bs = len(targets)
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device
        max_gt_num = max(num_gts)

        # use dummy contents to keep gradient map
        if self.num_denoising <= 0 or not self.training or max_gt_num == 0:
            dn_anchors = torch.full([bs, 0, 4], 0.5, device=device)
            dn_contents = torch.full([bs, 0, self.hidden_dim], self.num_classes, dtype=torch.int32, device=device)
            dn_mask = torch.ones([bs, 0], dtype=torch.bool, device=device)
            dn_meta = {
                "dn_positive_idx": [],
                "dn_num_group": -1,
                "dn_source": "none",
                "dummy_loss": (
                    self.denoising_class_embed(torch.full([bs, 1], self.num_classes, dtype=torch.int32, device=device))
                    * 0
                ).sum(),
            }

            return dn_anchors, dn_contents, dn_mask, dn_meta

        dn_anchors = torch.full([bs, max_gt_num, 4], 0.5, device=device)
        dn_contents = torch.full([bs, max_gt_num], self.num_classes, dtype=torch.int32, device=device)
        dn_mask = torch.ones([bs, max_gt_num], dtype=torch.bool, device=device)
        for i in range(bs):
            num_gt = num_gts[i]
            dn_contents[i, :num_gt] = targets[i]["labels"]
            # Convert GT boxes from valid region coordinates to global image coordinates
            gt_boxes = targets[i]["bbox"]
            if valid_ratios is not None:
                # Use the first level's valid ratio
                valid_ratio = valid_ratios[i, 0]  # [2] -> [w_ratio, h_ratio]
                gt_boxes = gt_boxes * torch.cat([valid_ratio, valid_ratio], dim=-1)[None, :]

            dn_anchors[i, :num_gt] = gt_boxes
            dn_mask[i, :num_gt] = False

        # each group has positive and negative queries.
        num_group = self.num_denoising // max_gt_num
        num_group = 1 if num_group == 0 else num_group
        dn_contents = dn_contents.tile([1, 2 * num_group])
        dn_anchors = dn_anchors.tile([1, 2 * num_group, 1])
        dn_mask = dn_mask.tile([1, 2 * num_group])
        dn_valid_mask = ~dn_mask
        # positive and negative mask
        negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
        negative_gt_mask[:, max_gt_num:] = 1
        negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
        positive_gt_mask = (1 - negative_gt_mask).squeeze(-1) * dn_valid_mask

        if self.label_noise_ratio > 0:
            mask = torch.rand_like(dn_contents, dtype=torch.float) < (self.label_noise_ratio * 0.5)
            # randomly put a new one here
            max_classes = self.num_classes if self.num_classes > 1 else self.num_classes + 1
            new_label = torch.randint_like(mask, 0, max_classes, dtype=dn_contents.dtype)
            dn_contents = torch.where(mask & dn_valid_mask, new_label, dn_contents)
        if self.box_noise_scale > 0:
            known_bbox = box_cxcywh_to_xyxy(dn_anchors)
            diff = torch.tile(dn_anchors[..., 2:] * 0.5, [1, 1, 2]) * self.box_noise_scale
            rand_sign = torch.randint_like(dn_anchors, 0, 2) * 2.0 - 1.0
            rand_part = torch.rand_like(dn_anchors)
            # [-w, +w] * scale for positive, 2 * [-w, +w] * scale for negative
            rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
            known_bbox += rand_sign * rand_part * diff
            known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
            dn_anchors = inverse_sigmoid(box_xyxy_to_cxcywh(known_bbox))

        dn_contents = self.denoising_class_embed(dn_contents)
        # contrastive denoising training positive index
        dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
        dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
        dn_meta = {"dn_positive_idx": dn_positive_idx, "dn_num_group": num_group, "dn_source": "gt", "dummy_loss": -1}
        return dn_anchors, dn_contents, dn_mask, dn_meta

    @TensorHook(keys=["main_logits", "main_mask", "main_boxes"], name="hqg_topk_source", switch="hqg_topk_source")
    def _get_detection_queries(self, features: torch.Tensor, features_mask: torch.Tensor, spatial_shapes: torch.Tensor):
        """
        Generate detection query inputs for the decoder.

        Args:
            features (torch.Tensor): [b,n,dim] Flattened multi-scale features
            features_mask (torch.Tensor): [b,n] Padding mask (True for padding tokens)
            spatial_shapes (torch.Tensor): [feats_layers,2] Spatial shapes (w,h) for each layer

        Returns:
            det_anchors (torch.Tensor): Detection query anchors in logits space (inverse_sigmoid format)
            det_contents (torch.Tensor): Detection query content embeddings
            det_mask (torch.Tensor): Detection query padding mask (True for padding queries)
            det_meta (Dict): Metadata containing TopK bboxes and logits for training alignment.
                             Note: bboxes in meta are normalized to valid image regions [0,1] for loss computation
                             (different coordinate system from anchors which are in logits space)
        """
        bs = features.shape[0]
        device = features.device
        det_meta = {
            "TopK_bboxes": [],
            "TopK_logits": [],
        }

        if self.mode == "learnable":
            # Learnable embeddings
            det_anchors = self.det_anchor.weight.unsqueeze(0).tile([bs, 1, 1])
            det_contents = self.det_content.weight.unsqueeze(0).tile([bs, 1, 1])
            # Learnable queries have no pre-decoder scores, so use a uniform placeholder for ROI weighting.
            main_logits = torch.zeros((bs, self.num_queries, 1), dtype=det_contents.dtype, device=device)
        elif self.mode == "generative":
            # Generative approach
            features_enc_out, feat_logits, feat_bbox = self._get_feature_score(features, features_mask, spatial_shapes)
            topK_det_contents, topK_det_logits, topK_det_anchors = self._select_topK(
                contents=features_enc_out, logits=feat_logits, anchors=feat_bbox, topk=self.num_queries  # bias+content
            )
            if self.training:
                det_meta["TopK_bboxes"].append(F.sigmoid(topK_det_anchors))
                det_meta["TopK_logits"].append(topK_det_logits)

            det_anchors = topK_det_anchors.detach()
            det_contents = topK_det_contents.detach()
            main_logits = topK_det_logits
        else:
            raise NotImplementedError("mode {} not supported".format(self.mode))

        det_mask = torch.zeros([bs, self.num_queries], dtype=torch.bool, device=device)
        main_mask = det_mask
        main_boxes = det_anchors.sigmoid()
        return det_anchors, det_contents, det_mask, det_meta

    def _get_feature_score(self, features: torch.Tensor, features_mask: torch.Tensor, spatial_shapes):
        if self.training or self.eval_spatial_size is None:
            det_anchors_base, det_valid_mask = self._generate_anchors(spatial_shapes, device=features.device)
        else:
            det_anchors_base = self.anchors
            det_valid_mask = self.valid_mask

        det_valid_mask = det_valid_mask * ~features_mask[:, :, None]
        features_enc_out = self.enc_output(det_valid_mask.to(features.dtype) * features)
        feat_logits = self.enc_score_head(features_enc_out)
        feat_bbox = self.enc_bbox_head(features_enc_out) + det_anchors_base
        return features_enc_out, feat_logits, feat_bbox

    @staticmethod
    def _build_attention_mask(num_dn: int, num_det: int, num_tracks: int, device: torch.device, dn_num_group=-1):
        # Build query self-attention mask
        query_nums = num_dn + num_det + num_tracks
        attn_mask = torch.full([query_nums, query_nums], False, dtype=torch.bool, device=device)
        # match query cannot see the Denoising
        if num_dn > 0:
            attn_mask[num_dn:, :num_dn] = True
        # DN cannot see each other (only apply when group info is valid)
        if num_dn > 0 and dn_num_group > 0:
            max_gt_num = num_dn // 2 // dn_num_group
            for i in range(dn_num_group):
                if i == 0:
                    attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1) : num_dn] = True
                elif i == dn_num_group - 1:
                    attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i] = True
                else:
                    attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1) : num_dn] = True
                    attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i] = True
        return attn_mask

    def forward(
        self,
        features: torch.Tensor,
        features_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        trackers: list[TrackInstances],
        valid_ratios: torch.Tensor,
        dn_targets: list[dict],
    ):
        """
        Generate all queries for decoder

        Returns:
            query_contents: [B, N_total, D] Combined query content embeddings
            query_anchors: [B, N_total, 4] Combined query anchor coordinates in logits space
            attn_mask: [N_total, N_total] Self-attention mask
            query_mask: [B, N_total] Query padding mask (True for padding)
            meta: Dict containing metadata for each query type
        """
        tracker_anchors, tracker_contents, tracker_mask = self._get_trackers(trackers)
        dn_anchors, dn_contents, dn_mask, dn_meta = self._get_denoising_queries(dn_targets, valid_ratios)
        det_anchors, det_contents, det_mask, det_meta = self._get_detection_queries(
            features, features_mask, spatial_shapes
        )
        meta = {
            "split": [dn_anchors.shape[1], dn_anchors.shape[1] + det_anchors.shape[1]],
            "query_nums": [dn_anchors.shape[1], det_anchors.shape[1], tracker_anchors.shape[1]],
            "dn_meta": dn_meta,
            "det_meta": det_meta,
        }
        # Combine all queries
        contents = torch.cat([dn_contents, det_contents, tracker_contents], dim=1)
        anchors = torch.cat([dn_anchors, det_anchors, tracker_anchors], dim=1)
        query_masks = torch.cat([dn_mask, det_mask, tracker_mask], dim=1)
        attn_mask = self._build_attention_mask(
            num_dn=dn_anchors.shape[1],
            num_det=det_anchors.shape[1],
            num_tracks=tracker_anchors.shape[1],
            dn_num_group=dn_meta["dn_num_group"],
            device=contents.device,
        )
        return contents, anchors, attn_mask, query_masks, meta
