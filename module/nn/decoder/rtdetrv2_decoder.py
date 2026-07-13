"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""
# Modified from RT-DETRv2 (https://github.com/zheli-hub/RT-DETRv2)

import copy
import inspect
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint
from typing import List

from utils.visualizer import TensorHook

# from .denoising import get_contrastive_denoising_training_group
from .hybrid_query_generator import HybridQueryGenerator
from .query_generator import QueryGenerator
from ..common import inverse_sigmoid, bias_init_with_prob, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from ..common import FFN, MLP, box_sine_embedding, FrozenBatchNorm2d
from ..common.transformer import MultiheadSelfAttention, build_ms_deformable_attention
from ..instance import TrackInstances


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation='relu',
        n_levels=4,
        n_points: int | List[int] = 4,
        cross_attn_method='default',
    ):
        super(TransformerDecoderLayer, self).__init__()

        # --- self attention
        self.self_attn = MultiheadSelfAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # --- cross attention
        self.cross_attn = build_ms_deformable_attention(
            embed_dim=d_model,
            num_heads=n_head,
            num_levels=n_levels,
            num_points=n_points,
            method=cross_attn_method,
            sigmoid_attn=False,
            visualize=False,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FFN(d_model, dim_feedforward, activation=activation, dropout=dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    # @TensorHook(["self_attn_weight", "query_pos_embed"])
    def forward(
        self,
        query,
        reference_points,
        feats,
        feats_spatial_shapes,
        attn_mask=None,
        feats_mask=None,
        query_padding_mask=None,
        query_pos_embed=None,
    ):
        # self attention
        query2, self_attn_weight = self.self_attn(
            x=query,
            pos=query_pos_embed,
            attn_mask=attn_mask,
            key_padding_mask=query_padding_mask,
            need_weights=True,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos_embed),
            reference_points=reference_points,
            value=feats,
            value_spatial_shapes=feats_spatial_shapes,
            value_mask=feats_mask,
        )
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # ffn
        query = self.ffn(query)
        return query


class TransformerDecoder(nn.Module):
    FOCUS_EVENT_NAME = "decoder_l0_det_query_focus"
    FOCUS_EVENT_SWITCH = "decoder_l0_det_query_focus"

    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        num_layers,
        eval_idx=-1,
        merge_det_track_layer=1,
        use_sine_pos=True,
        use_query_scale=False,
    ):
        super(TransformerDecoder, self).__init__()
        self.merge_det_track_layer = merge_det_track_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.use_sine_pos = use_sine_pos
        self.use_query_scale = use_query_scale

    def _build_query_pos_embed(self, ref_points_detach, query_pos_head):
        if self.use_sine_pos:
            anchor_embed = box_sine_embedding(pos=ref_points_detach, num_pos_feats=self.hidden_dim // 4)
            return query_pos_head(anchor_embed)
        return query_pos_head(ref_points_detach)

    @staticmethod
    def _freeze_track_ref_points(next_ref_bbox, ref_points_detach, det_end):
        return torch.cat((next_ref_bbox[:, :det_end, :], ref_points_detach[:, det_end:, :]), dim=1)

    def _refine_reference_points(
        self,
        inter_bias,
        ref_points_detach,
        ref_points_continuous,
        valid_ratios_4d,
        freeze_track_refs,
        det_end,
    ):
        inter_bias_global = inter_bias / valid_ratios_4d[:, None, :]
        base_box_detach = inverse_sigmoid(ref_points_detach)
        base_box_continuous = inverse_sigmoid(ref_points_continuous)

        next_ref_bbox = torch.sigmoid(base_box_detach + inter_bias_global)
        if freeze_track_refs:
            next_ref_bbox = self._freeze_track_ref_points(next_ref_bbox, ref_points_detach, det_end)

        train_ref_bbox = torch.sigmoid(base_box_continuous + inter_bias_global)
        return next_ref_bbox, train_ref_bbox

    def _collect_layer_outputs(
        self, layer_idx, output, next_ref_bbox, train_ref_bbox, score_head, dec_out_logits, dec_out_bboxes
    ):
        if self.training:
            dec_out_logits.append(score_head[layer_idx](output))
            dec_out_bboxes.append(train_ref_bbox)
        elif layer_idx == self.eval_idx:
            dec_out_logits.append(score_head[layer_idx](output))
            dec_out_bboxes.append(next_ref_bbox)

    @classmethod
    def _emit_decoder_focus_event(
        cls,
        layer,
        layer_idx,
        det_end,
        query_logits,
        query_boxes_before,
        query_boxes_after,
        query_mask,
    ):
        cross_attn = getattr(layer, "cross_attn", None)
        if cross_attn is None or not hasattr(cross_attn, "pop_last_visual_payload"):
            return

        payload = cross_attn.pop_last_visual_payload()
        if not isinstance(payload, dict):
            return

        sampling_locations = payload.get("sampling_locations", None)
        attention_weights = payload.get("attention_weights", None)
        value_spatial_shapes = payload.get("value_spatial_shapes", None)
        reference_points = payload.get("reference_points", None)
        if not all(isinstance(x, torch.Tensor) for x in (sampling_locations, attention_weights, query_logits, query_boxes_after)):
            return

        # Emit an explicitly named decoder-layer-0 det-query event so visual tasks do not have to infer call order.
        TensorHook.emit(
            name=cls.FOCUS_EVENT_NAME,
            switch=cls.FOCUS_EVENT_SWITCH,
            payload={
                "layer_idx": int(layer_idx),
                "det_end": int(det_end),
                "sampling_locations": sampling_locations[:, :det_end],
                "attention_weights": attention_weights[:, :det_end],
                "value_spatial_shapes": value_spatial_shapes,
                "reference_points": reference_points[:, :det_end],
                "query_logits": query_logits[:, :det_end],
                "query_boxes_before": query_boxes_before[:, :det_end],
                "query_boxes_after": query_boxes_after[:, :det_end],
                "query_mask": query_mask[:, :det_end] if isinstance(query_mask, torch.Tensor) else None,
            },
        )

    def forward(
        self,
        contents,
        anchors,
        features,
        features_spatial_shapes,
        valid_ratios,
        token_start_idx,
        bbox_head,
        score_head,
        query_pos_head,
        query_scale_head=None,
        attn_mask=None,
        query_mask=None,
        feats_padding_mask=None,
    ):
        dec_out_bboxes = []
        dec_out_logits = []
        queries = []
        ref_p = []

        det_end = token_start_idx[1]
        merge_from = self.merge_det_track_layer
        valid_ratios_4d = torch.cat([valid_ratios, valid_ratios], -1)
        valid_ratios_lvl0_4d = valid_ratios_4d[:, 0, :]
        det_attn_mask = attn_mask[:det_end, :det_end] if attn_mask is not None else None
        det_query_mask = query_mask[:, :det_end] if query_mask is not None else None

        output = contents
        ref_points_detach = torch.sigmoid(anchors)
        ref_points_continuous = ref_points_detach
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach[:, :, None] * valid_ratios_4d[:, None]
            merge_det_track = i >= merge_from
            query_pos_embed = self._build_query_pos_embed(ref_points_detach, query_pos_head)
            if merge_det_track and self.use_query_scale and query_scale_head is not None:
                query_pos_embed = query_scale_head(output) * query_pos_embed
            if merge_det_track:
                output = layer(
                    query=output,
                    query_pos_embed=query_pos_embed,
                    attn_mask=attn_mask,
                    query_padding_mask=query_mask,
                    feats=features,
                    feats_spatial_shapes=features_spatial_shapes,
                    feats_mask=feats_padding_mask,
                    reference_points=ref_points_input,
                )
            else:
                output_det = layer(
                    query=output[:, :det_end, :],
                    query_pos_embed=query_pos_embed[:, :det_end, :],
                    attn_mask=det_attn_mask,
                    query_padding_mask=det_query_mask,
                    feats=features,
                    feats_spatial_shapes=features_spatial_shapes,
                    feats_mask=feats_padding_mask,
                    reference_points=ref_points_input[:, :det_end, :, :],
                )
                output = torch.cat((output_det, output[:, det_end:, :]), dim=1)

            queries.append(output)

            # hack implementation for iterative bounding box refinement.
            inter_bias = bbox_head[i](output)  # logical offset in valid region coordinate system
            freeze_track_refs = i < merge_from
            next_ref_bbox, train_ref_bbox = self._refine_reference_points(
                inter_bias=inter_bias,
                ref_points_detach=ref_points_detach,
                ref_points_continuous=ref_points_continuous,
                valid_ratios_4d=valid_ratios_lvl0_4d,
                freeze_track_refs=freeze_track_refs,
                det_end=det_end,
            )

            if i == 0 and det_end > 0:
                focus_logits = score_head[i](output[:, :det_end, :])
                self._emit_decoder_focus_event(
                    layer=layer,
                    layer_idx=i,
                    det_end=det_end,
                    query_logits=focus_logits,
                    query_boxes_before=ref_points_detach,
                    query_boxes_after=next_ref_bbox,
                    query_mask=query_mask,
                )

            self._collect_layer_outputs(
                layer_idx=i,
                output=output,
                next_ref_bbox=next_ref_bbox,
                train_ref_bbox=train_ref_bbox,
                score_head=score_head,
                dec_out_logits=dec_out_logits,
                dec_out_bboxes=dec_out_bboxes,
            )

            ref_points_continuous = next_ref_bbox
            ref_points_detach = next_ref_bbox.detach()

            ref_p.append(ref_points_detach)
        ref_p = torch.stack(ref_p, dim=0)
        queries = torch.stack(queries, dim=0)

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), queries, ref_p


class RTDETRTransformerv2(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(
        self,
        num_classes=80,
        norm_style='BN',
        det_query_mode='generate',
        query_select_method='default',
        activation="relu",
        dropout=0.0,
        eps=1e-5,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        feat_levels=3,
        hidden_dim=256,
        num_points=4,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        use_det_dn=False,
        num_layers=6,
        nhead=8,
        num_queries=300,
        dim_feedforward=1024,
        merge_det_track_layer=1,
        cross_attn_method='default',
        eval_spatial_size=None,
        eval_idx=-1,
        aux_loss=True,
        use_sine_pos=True,
        use_query_scale=False,
        qpn_interact_max_state=1,
        hqg_init_feat_level=-1,
        hqg_init_feat_levels=None,
        hqg_num_learnable_memory=32,
        hqg_active_learnable_memory=None,
        hqg_num_blocks=3,
        hqg_det_dn_mask_track_memory=True,
    ):
        super().__init__()
        assert len(feat_channels) <= feat_levels
        assert query_select_method in (
            'default',
            'one2many',
            'agnostic',
        ), 'query_select_method should be "default" , "one2many" or "agnostic" but got {}'.format(query_select_method)
        assert cross_attn_method in (
            'default',
            'CUDA',
            'discrete',
        ), 'cross_attn_method should be "default" , "CUDA" or "discrete" but got {}'.format(cross_attn_method)
        assert box_noise_scale >= 0, 'box_noise_scale should be >0'
        assert label_noise_ratio >= 0, 'label_noise_ratio should be >=0'

        self.hidden_dim = hidden_dim
        self.nhead = nhead

        self.feat_strides = feat_strides
        self.feat_levels = feat_levels
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size

        self.aux_loss = aux_loss
        self.use_det_dn = use_det_dn
        self.use_sine_pos = use_sine_pos
        self.use_query_scale = use_query_scale
        self.det_dn_aux_last_only = False

        self.norm_style = norm_style
        self.cross_attn_method = cross_attn_method

        """ backbone feature projection and encoder """
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                            ('norm', self._build_norm(self.hidden_dim)),
                        ]
                    )
                )
            )
        input_channels_num = len(feat_channels)
        last_feat_channel = feat_channels[-1]
        for _ in range(self.feat_levels - input_channels_num):
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ('conv', nn.Conv2d(last_feat_channel, self.hidden_dim, 3, 2, padding=1, bias=False)),
                            ('norm', self._build_norm(self.hidden_dim)),
                        ]
                    )
                )
            )
            last_feat_channel = self.hidden_dim
            self.feat_strides.append(self.feat_strides[-1] * 2)

        """ Detection contents and anchors generate """
        # --- query generation mode
        self.det_query_mode = det_query_mode
        self.use_hybrid_qpn = det_query_mode == 'hybrid'
        if self.use_hybrid_qpn:
            self.query_generator = HybridQueryGenerator(
                hidden_dim=self.hidden_dim,
                ffn_dim=dim_feedforward,
                num_queries=num_queries,
                num_classes=num_classes,
                num_denoising=num_denoising,
                # Expose HQG depth for checkpoint-truncation ablations.
                num_blocks=hqg_num_blocks,
                num_learnable_memory=hqg_num_learnable_memory,
                use_dn=self.use_det_dn,
                eps=eps,
                qpn_interact_max_state=qpn_interact_max_state,
                init_feat_level=hqg_init_feat_level,
                init_feat_levels=hqg_init_feat_levels,
                active_learnable_memory=hqg_active_learnable_memory,
                det_dn_mask_track_memory=hqg_det_dn_mask_track_memory,
            )
        else:
            self.query_generator = QueryGenerator(
                hidden_dim=self.hidden_dim,
                num_queries=num_queries,
                num_classes=num_classes,
                mode=det_query_mode,
                query_select_method=query_select_method,
                num_denoising=num_denoising,
                label_noise_ratio=label_noise_ratio,
                box_noise_scale=box_noise_scale,
                use_det_dn=self.use_det_dn,
                feat_strides=feat_strides,
                eval_spatial_size=eval_spatial_size,
                eps=eps,
            )

        """ Transformer decoder module """
        # Input dimension depends on whether to use sinusoidal encoding, we suggest to use it like DAB-DETR style
        query_pos_dim = hidden_dim if self.use_sine_pos else 4
        self.query_pos_head = MLP(query_pos_dim, 2 * hidden_dim, hidden_dim, 2)
        self.query_scale = MLP(hidden_dim, hidden_dim, hidden_dim, 2) if self.use_query_scale else None
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)])

        # --- decoder layer
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            feat_levels,
            num_points,
            cross_attn_method=cross_attn_method,
        )
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            num_layers,
            eval_idx,
            merge_det_track_layer,
            use_sine_pos,
            use_query_scale,
        )
        """ Rest parameters """
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters in the order they are constructed"""
        bias = bias_init_with_prob(0.01)

        # 1. Backbone feature projection (input_proj)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)  # type: ignore

        # 2. Query position head
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        if self.query_scale is not None:
            init.xavier_uniform_(self.query_scale.layers[0].weight)
            init.xavier_uniform_(self.query_scale.layers[1].weight)

        # 3. Decoder heads (score and bbox for each layer)
        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)

    def _build_norm(self, *args, **kwargs):
        if self.norm_style == 'BN':
            return nn.BatchNorm2d(*args, **kwargs)
        elif self.norm_style == 'freeze_BN':
            return FrozenBatchNorm2d(*args, **kwargs)
        elif self.norm_style == 'LN':
            return nn.LayerNorm(*args, **kwargs)
        elif self.norm_style == 'GN':
            if len(args) == 1:
                args = (32,) + args
            return nn.GroupNorm(*args, **kwargs)  # type: ignore
        else:
            raise AttributeError(f"norm config {self.norm_style} not supported")

    def _get_encoder_input(self, feats: list[torch.Tensor], p_masks: list[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.feat_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.feat_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes_list = []
        mask_flatten = []
        for i, (feat, mask) in enumerate(zip(proj_feats, p_masks)):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            mask_flatten.append(mask.flatten(1))
            # Collect spatial shapes for tensor conversion
            spatial_shapes_list.append([h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        mask_flatten = torch.concat(mask_flatten, 1)
        spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long, device=feat_flatten.device)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in p_masks], 1)
        return feat_flatten, mask_flatten, spatial_shapes, valid_ratios

    def forward(self, feats, feats_mask, dn_targets, trackers, checkpoint_level):
        # input features projection and embedding
        feat, feats_masks, spatial_shapes, valid_ratios = self._get_encoder_input(feats, feats_mask)

        # Build unified decoder query inputs from the query generator.
        contents, anchors, attn_mask, query_mask, meta = self.query_generator(
            features=feat,
            features_mask=feats_masks,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            dn_targets=dn_targets,
            trackers=trackers,
        )
        decoder_query_mask = meta.get("decoder_query_mask", query_mask)

        # decoder
        if checkpoint_level > 1:
            out_bboxes, out_logits, out_queries, ref_p = checkpoint(
                self.decoder,
                contents,
                anchors,
                feat,
                spatial_shapes,
                valid_ratios,
                meta['split'],
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                self.query_scale,
                attn_mask,
                decoder_query_mask,
                feats_masks,
                use_reentrant=False,
            )  # type: ignore
        else:
            out_bboxes, out_logits, out_queries, ref_p = self.decoder(
                contents,
                anchors,
                feat,
                spatial_shapes,
                valid_ratios,
                meta['split'],
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                self.query_scale,
                attn_mask=attn_mask,
                query_mask=decoder_query_mask,
                feats_padding_mask=feats_masks,
            )

        # Output content integration
        split = meta['split']
        dn_end = split[0]
        dn_slice = slice(0, dn_end)
        main_slice = slice(dn_end, None)

        last_logits = out_logits[-1]
        last_bboxes = out_bboxes[-1]
        main_query_mask = query_mask[:, main_slice]
        norm_ratio = torch.concat([valid_ratios[:, 0, :], valid_ratios[:, 0, :]], dim=1)

        out = {
            'main_outputs': {
                'pred_logits': last_logits[:, main_slice, :],
                'pred_bboxes': last_bboxes[:, main_slice, :],
                'query_mask': main_query_mask,
            },
            'pred_queries': out_queries[:, :, main_slice, :],
            'pred_ref_point': ref_p[:, :, main_slice, :],
            'meta': {
                'split': split,
                'query_nums': meta['query_nums'],
                'dn_meta': meta['dn_meta'],
                'ratios': norm_ratio,
                'qpn_aux_unmatched': bool(self.use_hybrid_qpn),
            },
        }

        if self.training:
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(
                    out_logits[:-1, :, main_slice, :], out_bboxes[:-1, :, main_slice, :], main_query_mask
                )
                out['qpn_aux_outputs'] = self._set_aux_loss(
                    meta['det_meta']["TopK_logits"], meta['det_meta']["TopK_bboxes"]
                )
            qpn_main_dense = meta.get("det_meta", {}).get("main_dense", None)
            if qpn_main_dense is not None:
                out["qpn_main_dense_outputs"] = qpn_main_dense
            if dn_end > 0:
                dn_branch = meta.get("det_meta", {}).get("dn_branch", None)
                if dn_branch is not None:
                    out["qpn_dn_outputs"] = {
                        "pred_logits": dn_branch["TopK_logits"],
                        "pred_bboxes": dn_branch["TopK_bboxes"],
                        "query_mask": dn_branch["TopK_mask"],
                    }

                # middle-stage training can keep DN alive while restricting det_dn supervision to the last head.
                if self.aux_loss and not self.det_dn_aux_last_only:
                    out["det_dn_outputs"] = self._set_aux_loss(
                        out_logits[:, :, dn_slice, :], out_bboxes[:, :, dn_slice, :], query_mask[:, dn_slice]
                    )
                else:
                    out["det_dn_outputs"] = {
                        "pred_logits": last_logits[:, dn_slice, :],
                        "pred_bboxes": last_bboxes[:, dn_slice, :],
                        "query_mask": query_mask[:, dn_slice],
                    }

        return out

    @staticmethod
    def get_valid_ratio(mask):
        """
        Args:
            mask: NestedTensor's mask

        Returns:

        """
        _, H, W = mask.shape  # (B, H, W)
        valid_H = torch.sum(~mask[:, :, 0], 1)  # (B, )
        valid_W = torch.sum(~mask[:, 0, :], 1)  # (B, )
        valid_ratio_h = valid_H.float() / H  # (B, )
        valid_ratio_w = valid_W.float() / W  # (B, )
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio  # (B, 2)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, query_mask=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if query_mask is None:
            res = [{'pred_logits': a, 'pred_bboxes': b} for a, b in zip(outputs_class, outputs_coord)]
        else:
            res = [
                {'pred_logits': a, 'pred_bboxes': b, 'query_mask': query_mask}
                for a, b in zip(outputs_class, outputs_coord)
            ]
        return res


def build(config):
    # load backbone
    sig = inspect.signature(RTDETRTransformerv2)
    _cfg = {k: v for k, v in config['Decoder'].items() if k in sig.parameters}
    model = RTDETRTransformerv2(**_cfg)
    return model
