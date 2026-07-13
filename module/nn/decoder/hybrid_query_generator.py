from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from ..instance import TrackInstances
from ..life_cycle_management import TrackState
from ..common import FFN, MLP, bias_init_with_prob
from ..common.transformer import CrossAttention, MultiheadSelfAttention
from ..common.spatiotemporal_embedding import RandomFourierEncoder, SpatioTemporalEmbedding
from utils.visualizer import TensorHook


class HybridQPNBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiheadSelfAttention(hidden_dim, num_heads, dropout=dropout)
        self.cross_attn = CrossAttention(
            embedding_dim=hidden_dim,
            num_heads=num_heads,
            downsample_rate=1,
            dropout=dropout,
            kv_in_dim=hidden_dim,
        )
        self.ffn = FFN(d_model=hidden_dim, dim_feedforward=dim_feedforward, activation="gelu", dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None,
        tgt_pos: torch.Tensor,
        mem_pos: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sa_out, _ = self.self_attn(x=x, pos=tgt_pos, key_padding_mask=tgt_mask, need_weights=False)
        x = self.norm1(x + sa_out)

        q = x + tgt_pos
        k = memory + mem_pos
        ca_out = self.cross_attn(q, k, memory, memory_mask, attn_mask)
        x = self.norm2(x + ca_out)

        x = self.ffn(x)
        return x


class HybridQueryGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        ffn_dim: int = 1024,
        num_queries: int = 300,
        num_classes: int = 80,
        num_denoising: int = 100,
        num_blocks: int = 3,
        num_heads: int = 8,
        num_learnable_memory: int = 32,
        num_temporal_bins: int = 32,
        use_dn: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-5,
        qpn_interact_max_state: int = 1,
        init_feat_level: int = -1,
        init_feat_levels: list[int] | tuple[int, ...] | None = None,
        active_learnable_memory: int | None = None,
        det_dn_mask_track_memory: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_denoising = num_denoising
        self.num_learnable_memory = int(num_learnable_memory)
        self.use_dn = use_dn
        self.eps = eps
        self.qpn_interact_max_state = int(qpn_interact_max_state)
        self.init_feat_level = int(init_feat_level)
        self.init_feat_levels = self._normalize_init_feat_levels(init_feat_level, init_feat_levels)
        self.active_learnable_memory = self._resolve_active_learnable_memory(active_learnable_memory)
        self.det_dn_mask_track_memory = bool(det_dn_mask_track_memory)

        self.enc_output = nn.Sequential(
            OrderedDict(
                [
                    ("proj", nn.Linear(hidden_dim, hidden_dim)),
                    ("norm", nn.LayerNorm(hidden_dim)),
                ]
            )
        )
        self.layers = nn.ModuleList(
            [HybridQPNBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        self.feats_pos_encoder = RandomFourierEncoder(input_dim=4, hidden_dim=hidden_dim)
        self.st_embedding = SpatioTemporalEmbedding(
            hidden_dim=hidden_dim,
            num_temporal_bins=num_temporal_bins,
            spatial_encoder=self.feats_pos_encoder,
        )

        self.learnable_memory = nn.Parameter(torch.zeros(num_learnable_memory, hidden_dim))
        self.learnable_pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.learnable_time_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.register_buffer(
            "learnable_mask",
            torch.zeros(1, self.num_learnable_memory, dtype=torch.bool),
            persistent=False,
        )
        if self.active_learnable_memory < self.num_learnable_memory:
            generator = torch.Generator().manual_seed(torch.initial_seed())
            active_indices = (
                torch.randperm(self.num_learnable_memory, generator=generator)[
                    : self.active_learnable_memory
                ]
                .sort()
                .values.tolist()
            )
        else:
            active_indices = list(range(self.num_learnable_memory))
        # Keep HQG active indices as a plain Python attribute so DDP does not
        # treat this inference-time subset selector as a mutable buffer involved in autograd.
        self.active_learnable_indices = tuple(int(idx) for idx in active_indices)

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        nn.init.xavier_uniform_(self.enc_output[0].weight)
        nn.init.constant_(self.enc_output[0].bias, 0.0)
        nn.init.constant_(self.enc_score_head.bias, bias)
        nn.init.constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.enc_bbox_head.layers[-1].bias, 0.0)

        nn.init.normal_(self.learnable_memory, std=0.02)
        nn.init.normal_(self.learnable_pos_embed, std=0.02)
        nn.init.constant_(self.learnable_time_embed, 0.0)

    def stopDN(self) -> bool:
        """
        Stop DN branch generation at runtime.

        Returns:
            bool: True if any DN-related flag/state was changed in this call.
        """
        changed = False
        if bool(getattr(self, "use_dn", False)):
            self.use_dn = False
            changed = True
        if hasattr(self, "use_det_dn") and bool(getattr(self, "use_det_dn")):
            self.use_det_dn = False
            changed = True
        if int(getattr(self, "num_denoising", 0)) != 0:
            self.num_denoising = 0
            changed = True
        self._dn_stopped = True
        return changed

    @staticmethod
    def _normalize_init_feat_levels(
        init_feat_level: int,
        init_feat_levels: int | list[int] | tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if init_feat_levels is None:
            return (int(init_feat_level),)
        if isinstance(init_feat_levels, int):
            return (int(init_feat_levels),)
        if len(init_feat_levels) == 0:
            raise ValueError("HybridQueryGenerator init_feat_levels must not be empty.")

        levels = tuple(int(x) for x in init_feat_levels)
        if len(set(levels)) != len(levels):
            raise ValueError(f"HybridQueryGenerator init_feat_levels has duplicates: {levels}")
        return levels

    def _resolve_active_learnable_memory(self, active_learnable_memory: int | None) -> int:
        if self.num_learnable_memory < 0:
            raise ValueError(f"num_learnable_memory must be >= 0, got {self.num_learnable_memory}")

        if active_learnable_memory is None:
            return self.num_learnable_memory

        active = int(active_learnable_memory)
        if active < 0:
            raise ValueError(f"active_learnable_memory must be >= 0, got {active}")
        if active > self.num_learnable_memory:
            raise ValueError(
                f"active_learnable_memory={active} exceeds num_learnable_memory={self.num_learnable_memory}"
            )
        return active

    def _generate_anchors(self, spatial_shapes: torch.Tensor, device: torch.device, grid_size: float = 0.05):
        if spatial_shapes.ndim != 2 or spatial_shapes.shape[0] < 1 or spatial_shapes.shape[1] != 2:
            raise ValueError(
                "HybridQueryGenerator expects spatial_shapes as [num_levels, 2], "
                f"got {tuple(spatial_shapes.shape)}"
            )

        anchors = []
        for lvl_shape in spatial_shapes:
            h_i = int(lvl_shape[0].item())
            w_i = int(lvl_shape[1].item())
            yy, xx = torch.meshgrid(torch.arange(h_i, device=device), torch.arange(w_i, device=device), indexing="ij")
            xy = torch.stack([xx, yy], dim=-1)
            xy = (xy.unsqueeze(0) + 0.5) / torch.tensor([w_i, h_i], dtype=torch.float32, device=device)
            wh = torch.full_like(xy, grid_size)
            anchors.append(torch.cat([xy, wh], dim=-1).reshape(1, h_i * w_i, 4))

        anchors = torch.cat(anchors, dim=1)
        anchors = torch.logit(anchors.clamp(self.eps, 1 - self.eps))
        return anchors

    def _get_track_mem(
        self,
        trackers: list[TrackInstances],
        batch_size: int,
        device: torch.device,
        max_state_inclusive: int | None = None,
        extra_states: tuple[int, ...] | None = None,
    ):
        if len(trackers) != batch_size:
            raise ValueError(f"trackers length mismatch: {len(trackers)} != {batch_size}")
        keep_counts: list[int] = []
        for track in trackers:
            if len(track) == 0:
                keep_counts.append(0)
                continue
            states = getattr(track, "states", None)
            if states is None:
                keep_counts.append(len(track))
                continue
            state_vals = states.to(device=device, dtype=torch.long)[: len(track)]
            keep_mask = torch.ones_like(state_vals, dtype=torch.bool)
            if max_state_inclusive is not None:
                keep_mask = keep_mask & (state_vals <= int(max_state_inclusive))
            if extra_states is not None and len(extra_states) > 0:
                extra_state_tensor = torch.as_tensor(extra_states, dtype=torch.long, device=device)
                extra_mask = (state_vals[:, None] == extra_state_tensor[None, :]).any(dim=1)
                keep_mask = keep_mask | extra_mask
            keep_counts.append(int(keep_mask.sum().item()))

        max_tracks = max(keep_counts, default=0)
        if max_tracks == 0:
            anchors = torch.zeros((batch_size, 0, 4), dtype=torch.float32, device=device)
            contents = torch.zeros((batch_size, 0, self.hidden_dim), dtype=torch.float32, device=device)
            pad_mask = torch.ones((batch_size, 0), dtype=torch.bool, device=device)
            # Keep the temporal embedding graph-connected even when no track memory exists on this rank.
            pos = contents + self.st_embedding.temporal_embed.weight.sum() * 0.0
            return anchors.detach(), contents, pad_mask, pos

        anchors = torch.zeros((batch_size, max_tracks, 4), dtype=torch.float32, device=device)
        contents = torch.zeros((batch_size, max_tracks, self.hidden_dim), dtype=torch.float32, device=device)
        t_pos_idx = torch.zeros((batch_size, max_tracks), dtype=torch.long, device=device)
        pad_mask = torch.ones((batch_size, max_tracks), dtype=torch.bool, device=device)

        for b, track in enumerate(trackers):
            if len(track) == 0:
                continue
            states = getattr(track, "states", None)
            if states is None:
                keep_idx = torch.arange(len(track), dtype=torch.long, device=device)
            else:
                state_vals = states.to(device=device, dtype=torch.long)[: len(track)]
                keep_mask = torch.ones_like(state_vals, dtype=torch.bool)
                if max_state_inclusive is not None:
                    keep_mask = keep_mask & (state_vals <= int(max_state_inclusive))
                if extra_states is not None and len(extra_states) > 0:
                    extra_state_tensor = torch.as_tensor(extra_states, dtype=torch.long, device=device)
                    extra_mask = (state_vals[:, None] == extra_state_tensor[None, :]).any(dim=1)
                    keep_mask = keep_mask | extra_mask
                keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
            n = int(keep_idx.numel())
            if n == 0:
                continue
            pad_mask[b, :n] = False
            anchors[b, :n] = track.ref_pts.to(device=device, dtype=torch.float32).index_select(0, keep_idx)
            contents[b, :n] = track.query_embed.to(device=device, dtype=torch.float32).index_select(0, keep_idx)

            disappear_t = getattr(track, "disappear_time", None)
            if disappear_t is None:
                dis = torch.zeros(n, dtype=torch.long, device=device)
            else:
                dis = disappear_t.to(device=device).index_select(0, keep_idx).to(torch.long).clamp_min(0)
            t_pos_idx[b, :n] = dis

        valid = ~pad_mask
        boxes = anchors.sigmoid().clamp(0.0, 1.0)
        pos = self.st_embedding(boxes=boxes, t_pos_idx=t_pos_idx, valid_mask=valid)
        return anchors.detach(), contents, pad_mask, pos

    def _encode_branch(self, tokens: torch.Tensor, token_mask: torch.Tensor, anchors_base: torch.Tensor):
        valid_tokens = ~token_mask.unsqueeze(-1)
        encoded = self.enc_output(tokens * valid_tokens.to(tokens.dtype))
        logits = self.enc_score_head(encoded)
        bbox = self.enc_bbox_head(encoded) + anchors_base
        return encoded, logits, bbox

    @staticmethod
    def _gather_topk(contents: torch.Tensor, logits: torch.Tensor, anchors: torch.Tensor, k: int):
        n = logits.shape[1]
        if n == 0 or k <= 0:
            b = logits.shape[0]
            return (
                contents.new_zeros((b, 0, contents.shape[-1])),
                logits.new_zeros((b, 0, logits.shape[-1])),
                anchors.new_zeros((b, 0, anchors.shape[-1])),
            )

        k = min(k, n)
        scores = logits.squeeze(-1) if logits.shape[-1] == 1 else logits.max(-1).values
        topk_idx = scores.topk(k, dim=-1).indices

        def gather(x: torch.Tensor):
            return x.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        return gather(contents), gather(logits), gather(anchors)

    @staticmethod
    def _pad_to_len(contents: torch.Tensor, logits: torch.Tensor, anchors: torch.Tensor, target_len: int):
        b, cur, d = contents.shape
        mask = torch.zeros((b, target_len), dtype=torch.bool, device=contents.device)
        if cur >= target_len:
            return contents[:, :target_len], logits[:, :target_len], anchors[:, :target_len], mask

        pad = target_len - cur
        mask[:, cur:] = True
        contents = torch.cat([contents, contents.new_zeros((b, pad, d))], dim=1)
        logits = torch.cat([logits, logits.new_full((b, pad, logits.shape[-1]), -1e8)], dim=1)
        anchors = torch.cat([anchors, anchors.new_zeros((b, pad, anchors.shape[-1]))], dim=1)
        return contents, logits, anchors, mask

    @staticmethod
    def _build_attention_mask(num_dn: int, num_det: int, num_tracks: int, device: torch.device):
        total = num_dn + num_det + num_tracks
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        if num_dn > 0:
            mask[:num_dn, num_dn:] = True
            mask[num_dn:, :num_dn] = True
        return mask

    @staticmethod
    def _build_decoder_query_mask(query_mask: torch.Tensor, num_dn: int) -> torch.Tensor:
        if num_dn <= 0 or query_mask.shape[1] < num_dn:
            return query_mask
        empty_dn_batch = query_mask[:, :num_dn].all(dim=1)
        if not torch.any(empty_dn_batch):
            return query_mask

        decoder_query_mask = query_mask.clone()
        # Keep empty DN slots masked for loss/matching, but let decoder self-attention
        # see them as isolated zero tokens so per-sample DN blocks never become all-masked rows.
        decoder_query_mask[empty_dn_batch, :num_dn] = False
        return decoder_query_mask

    def _resolve_init_level_indices(self, num_levels: int) -> tuple[int, ...]:
        resolved: list[int] = []
        for raw_idx in self.init_feat_levels:
            level_idx = int(raw_idx)
            if level_idx < 0:
                level_idx += num_levels
            if level_idx < 0 or level_idx >= num_levels:
                raise ValueError(
                    f"HybridQueryGenerator init_feat_levels={self.init_feat_levels} "
                    f"contains out-of-range index {raw_idx} for {num_levels} levels"
                )
            resolved.append(level_idx)
        if len(set(resolved)) != len(resolved):
            raise ValueError(
                f"HybridQueryGenerator init_feat_levels={self.init_feat_levels} "
                f"resolves to duplicated levels {tuple(resolved)}"
            )
        return tuple(resolved)

    def _select_init_tokens(
        self,
        features: torch.Tensor,
        features_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_levels = int(spatial_shapes.shape[0])
        level_indices = self._resolve_init_level_indices(num_levels)
        level_lens = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(dtype=torch.long, device=features.device)
        level_ends = torch.cumsum(level_lens, dim=0)
        feat_slices = []
        mask_slices = []
        shape_slices = []
        for level_idx in level_indices:
            level_end = int(level_ends[level_idx].item())
            level_len = int(level_lens[level_idx].item())
            level_start = level_end - level_len
            if features.shape[1] < level_end or features_mask.shape[1] < level_end:
                raise ValueError(
                    "HybridQueryGenerator init feature slice exceeds flattened features: "
                    f"level={level_idx}, need_end={level_end}, feats={features.shape[1]}, mask={features_mask.shape[1]}"
                )

            feat_slices.append(features[:, level_start:level_end, :])
            mask_slices.append(features_mask[:, level_start:level_end])
            shape_slices.append(spatial_shapes[level_idx : level_idx + 1])

        # Concatenate the configured HQG seed levels in yaml order so ablations can
        # compare single-level and multi-level image priors without changing encoder flattening.
        init_feats = torch.cat(feat_slices, dim=1)
        init_mask = torch.cat(mask_slices, dim=1)
        init_shapes = torch.cat(shape_slices, dim=0).to(device=features.device)
        return init_feats, init_mask, init_shapes

    @TensorHook(keys=["main_logits", "main_mask", "main_boxes"], name="hqg_topk_source", switch="hqg_topk_source")
    def forward(
        self,
        features: torch.Tensor,
        features_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        trackers: list[TrackInstances],
        valid_ratios: torch.Tensor,
        dn_targets: list[dict],
    ):
        if trackers is None:
            raise ValueError("HybridQueryGenerator requires trackers.")
        if spatial_shapes.ndim != 2 or spatial_shapes.shape[0] < 1:
            raise ValueError(f"Invalid spatial_shapes shape: {tuple(spatial_shapes.shape)}")

        b = features.shape[0]
        device = features.device
        low_feats, low_mask, low_shapes = self._select_init_tokens(features, features_mask, spatial_shapes)

        low_anchors = self._generate_anchors(low_shapes, device=device)
        low_pos_feat = low_anchors.sigmoid().clamp(0.0, 1.0)
        low_pos = self.feats_pos_encoder(low_pos_feat).expand(b, -1, -1)

        recover_inter_states = (
            (int(TrackState.RECOVER),)
            if self.qpn_interact_max_state is not None and int(self.qpn_interact_max_state) >= 0
            else None
        )
        # RECOVER is weak-positive track context for det-DN/HQG, while
        # ordinary LOST tracks still stay out of the interaction memory.
        _, inter_mem, inter_mask, inter_pos = self._get_track_mem(
            trackers,
            b,
            device,
            max_state_inclusive=self.qpn_interact_max_state,
            extra_states=recover_inter_states,
        )
        t_anchors, t_mem, t_mask, _ = self._get_track_mem(trackers, b, device)

        active_learnable_memory = self.active_learnable_memory
        # All HQG blocks share this subset, sampled once during model initialization.
        learn_pos = (self.learnable_pos_embed + self.learnable_time_embed).expand(b, active_learnable_memory, -1)
        learn_mask = self.learnable_mask[:, :active_learnable_memory].expand(b, -1)
        active_learnable_indices = torch.as_tensor(
            self.active_learnable_indices,
            device=self.learnable_memory.device,
            dtype=torch.long,
        )
        learn_main = self.learnable_memory.index_select(0, active_learnable_indices).unsqueeze(0).expand(b, -1, -1)
        learn_dn = learn_main

        main_mem = torch.cat([inter_mem, learn_main], dim=1)
        main_mem_pos = torch.cat([inter_pos, learn_pos], dim=1)
        main_mem_mask = torch.cat([inter_mask, learn_mask], dim=1)

        run_dn = self.training and self.use_dn and self.num_denoising > 0
        has_track = (~inter_mask).any(dim=1) if run_dn else torch.zeros((b,), dtype=torch.bool, device=device)
        dn_src_idx = torch.arange(b, dtype=torch.long, device=device) if run_dn else torch.empty((0,), dtype=torch.long, device=device)
        dn_batch_pairs: list[list[int]] = []

        if run_dn:
            dn_batch_pairs = [[int(s), int(b + i)] for i, s in enumerate(dn_src_idx.tolist())]
            dnf = low_feats.index_select(0, dn_src_idx)
            dnm = low_mask.index_select(0, dn_src_idx)
            dnp = low_pos.index_select(0, dn_src_idx)

            # DN: simulate tracker-empty detection. The track-memory view can be masked or
            # preserved to support auxiliary-bypass ablations from yaml.
            cnt = dn_src_idx.numel()
            lm = learn_dn.index_select(0, dn_src_idx)
            lp = learn_pos.index_select(0, dn_src_idx)
            lmask = learn_mask.index_select(0, dn_src_idx)
            if self.det_dn_mask_track_memory:
                ts = inter_mem.shape[1]
                t0 = inter_mem.new_zeros((cnt, ts, self.hidden_dim))
                p0 = inter_pos.new_zeros((cnt, ts, self.hidden_dim))
                m0 = torch.ones((cnt, ts), dtype=torch.bool, device=device)
                dn_mem = torch.cat([t0, lm], dim=1)
                dn_mem_pos = torch.cat([p0, lp], dim=1)
                dn_mem_mask = torch.cat([m0, lmask], dim=1)
            else:
                dn_track_mem = inter_mem.index_select(0, dn_src_idx)
                dn_track_pos = inter_pos.index_select(0, dn_src_idx)
                dn_track_mask = inter_mask.index_select(0, dn_src_idx)
                dn_mem = torch.cat([dn_track_mem, lm], dim=1)
                dn_mem_pos = torch.cat([dn_track_pos, lp], dim=1)
                dn_mem_mask = torch.cat([dn_track_mask, lmask], dim=1)

            low_feats = torch.cat([low_feats, dnf], dim=0)
            low_mask = torch.cat([low_mask, dnm], dim=0)
            low_pos = torch.cat([low_pos, dnp], dim=0)
            mem = torch.cat([main_mem, dn_mem], dim=0)
            mem_pos = torch.cat([main_mem_pos, dn_mem_pos], dim=0)
            mem_mask = torch.cat([main_mem_mask, dn_mem_mask], dim=0)
        else:
            mem = main_mem
            mem_pos = main_mem_pos
            mem_mask = main_mem_mask

        tokens = low_feats
        for layer in self.layers:
            tokens = layer(tokens, low_mask, mem, mem_mask, low_pos, mem_pos)
        main_tokens = tokens[:b]
        dn_tokens = tokens[b:] if run_dn else None

        main_enc, main_logits_all, main_bbox_all = self._encode_branch(main_tokens, low_mask[:b], low_anchors)

        main_contents, main_logits, main_anchors = self._gather_topk(
            main_enc, main_logits_all, main_bbox_all, self.num_queries
        )
        main_contents, main_logits, main_anchors, main_mask = self._pad_to_len(
            main_contents, main_logits, main_anchors, self.num_queries
        )
        # Expose normalized HQG TopK boxes directly so visual tasks can render ROI maps in inference.
        main_boxes = main_anchors.sigmoid()

        if dn_tokens is None:
            dn_anchors = main_anchors.new_zeros((b, 0, 4))
            dn_contents = main_contents.new_zeros((b, 0, self.hidden_dim))
            dn_logits = main_logits.new_zeros((b, 0, self.num_classes))
            dn_mask = torch.ones((b, 0), dtype=torch.bool, device=device)
            dn_meta = {
                "dn_source": "none",
                "dn_num_group": 0,
                "dn_positive_idx": [],
                "dn_batch_pairs": [],
                "dn_src_idx": [],
                "dummy_loss": None,
            }
        else:
            dn_enc, dn_logits_all, dn_bbox_all = self._encode_branch(dn_tokens, dnm, low_anchors)
            dn_contents_sub, dn_logits_sub, dn_anchors_sub = self._gather_topk(
                dn_enc, dn_logits_all, dn_bbox_all, self.num_denoising
            )
            dn_contents_sub, dn_logits_sub, dn_anchors_sub, dn_mask_sub = self._pad_to_len(
                dn_contents_sub, dn_logits_sub, dn_anchors_sub, self.num_denoising
            )
            dn_anchors = main_anchors.new_zeros((b, self.num_denoising, 4))
            dn_contents = main_contents.new_zeros((b, self.num_denoising, self.hidden_dim))
            dn_logits = main_logits.new_zeros((b, self.num_denoising, self.num_classes))
            dn_mask = torch.ones((b, self.num_denoising), dtype=torch.bool, device=device)

            dn_anchors[dn_src_idx] = dn_anchors_sub
            dn_contents[dn_src_idx] = dn_contents_sub
            dn_logits[dn_src_idx] = dn_logits_sub
            dn_mask[dn_src_idx] = dn_mask_sub
            dn_mask[~has_track] = True

            dn_meta = {
                "dn_source": "det_dn",
                "dn_num_group": 0,
                "dn_positive_idx": [],
                "dn_batch_pairs": dn_batch_pairs,
                "dn_src_idx": torch.nonzero(has_track, as_tuple=False).squeeze(-1).detach().cpu().tolist(),
                "dummy_loss": None,
            }

        det_meta: dict[str, Any] = {
            "TopK_bboxes": [],
            "TopK_logits": [],
            "qpn_aux_match_types": [],
        }
        if self.training:
            det_meta["TopK_bboxes"].append(main_boxes)
            det_meta["TopK_logits"].append(main_logits)
            det_meta["qpn_aux_match_types"].append("unmatched_gt")
            det_meta["main_dense"] = {
                "pred_logits": main_logits_all,
                "pred_bboxes": main_bbox_all.sigmoid(),
                "query_mask": low_mask[:b],
            }

            if dn_tokens is not None:
                det_meta["dn_branch"] = {
                    "TopK_bboxes": dn_anchors.sigmoid(),
                    "TopK_logits": dn_logits,
                    "TopK_mask": dn_mask,
                }

        combined_contents = torch.cat([dn_contents.detach(), main_contents.detach(), t_mem], dim=1)
        combined_anchors = torch.cat([dn_anchors.detach(), main_anchors.detach(), t_anchors], dim=1)
        combined_mask = torch.cat([dn_mask, main_mask, t_mask], dim=1)

        query_nums = [dn_anchors.shape[1], main_anchors.shape[1], t_anchors.shape[1]]
        split = [query_nums[0], query_nums[0] + query_nums[1]]
        decoder_query_mask = self._build_decoder_query_mask(combined_mask, query_nums[0])
        attn_mask = self._build_attention_mask(
            num_dn=query_nums[0],
            num_det=query_nums[1],
            num_tracks=query_nums[2],
            device=device,
        )

        meta = {
            "split": split,
            "query_nums": query_nums,
            "decoder_query_mask": decoder_query_mask,
            "dn_meta": dn_meta,
            "det_meta": det_meta,
        }
        return combined_contents, combined_anchors, attn_mask, combined_mask, meta
