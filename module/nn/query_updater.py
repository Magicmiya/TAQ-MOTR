# Copyright (c) Ruopeng Gao. All Rights Reserved.
# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)
import inspect
from typing import List

import torch
import torch.nn as nn

from module.nn.common import FFN, MLP
from module.nn.common import box_sine_embedding, inverse_sigmoid
from .instance import TrackInstances
from .life_cycle_management import TrackState


class QueryUpdater(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        tp_drop_ratio: float,
        fp_insert_ratio: float,
        dropout: float,
        long_memory_lambda: float,
        update_max_state: int = 1,
        no_tracking_augment: bool = True,
        visualize: bool = False,
        use_sine_pos: bool = True,
    ):
        super(QueryUpdater, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
        self.dropout = dropout
        self.long_memory_lambda = long_memory_lambda
        self.update_max_state = int(update_max_state)
        self.no_tracking_augment = no_tracking_augment

        self.visualize = visualize
        self.use_sine_pos = use_sine_pos

        # Net
        self.pos_embedding = box_sine_embedding if use_sine_pos else lambda a, b: a
        self.confidence_weight_net = nn.Sequential(
            MLP(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_layers=2,
            ),
            nn.Sigmoid(),
        )
        self.short_memory_fusion = MLP(
            input_dim=2 * self.hidden_dim,
            hidden_dim=2 * self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2,
        )
        self.memory_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        self.memory_dropout = nn.Dropout(self.dropout)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_ffn = FFN(self.hidden_dim, self.ffn_dim, dropout=self.dropout)
        self.query_feat_dropout = nn.Dropout(self.dropout)
        self.query_feat_norm = nn.LayerNorm(self.hidden_dim)
        self.query_feat_ffn = FFN(self.hidden_dim, self.ffn_dim, dropout=self.dropout)
        # Input dimension depends on whether to use sinusoidal encoding
        query_pos_input_dim = self.hidden_dim * 2 if use_sine_pos else 4
        self.query_pos_head = MLP(
            input_dim=query_pos_input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2,
        )

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tracks: List[TrackInstances]):
        return self.update_tracks_embedding(tracks)

    def update_tracks_embedding(self, tracks: List[TrackInstances]):
        updated_tracks: List[TrackInstances] = []
        for track in tracks:
            if len(track) > 0:
                updated_track = track.clone()
                track_states = getattr(track, "states", None)
                if track_states is None:
                    is_pos = torch.ones((len(track),), dtype=torch.bool, device=track.device)
                else:
                    state_vals = track_states.to(device=track.device, dtype=torch.long)[: len(track)]
                    # NEW/RELIABLE refresh positive memory, while NOISE
                    # remains a one-frame hard-negative update. RECOVER is a
                    # weak GT-matched query for HQG/det-DN only and must not
                    # refresh QueryUpdater memory.
                    is_pos = (state_vals <= self.update_max_state) | (state_vals == int(TrackState.NOISE))
                updated_ref_pts = track.ref_pts.clone()
                updated_ref_pts[is_pos] = inverse_sigmoid(track.boxes[is_pos].detach().clone())

                # Generate position embedding based on configuration
                query_pos = self.pos_embedding(updated_ref_pts.sigmoid(), self.hidden_dim // 2)
                output_embed = track.output_embed
                last_output_embed = track.last_output
                long_memory = track.long_memory.detach()

                # Confidence Weight
                confidence_weight = self.confidence_weight_net(output_embed)

                # Adaptive Aggregation
                short_memory = self.short_memory_fusion(torch.cat((confidence_weight * output_embed, last_output_embed), dim=-1))

                # Query Feature Generate
                query_pos = self.query_pos_head(query_pos)
                q, k, tgt = (
                    short_memory + query_pos,
                    long_memory + query_pos,
                    output_embed,
                )

                tgt2 = self.memory_attn(q[None, :], k[None, :], tgt[None, :])[0][0, :]
                tgt = tgt + self.memory_dropout(tgt2)
                tgt = self.memory_norm(tgt)
                tgt = self.memory_ffn(tgt)
                # Long Memory ResNet
                query_feat = long_memory + self.query_feat_dropout(tgt)
                query_feat = self.query_feat_norm(query_feat)
                query_feat = self.query_feat_ffn(query_feat)

                long_memory = (1 - self.long_memory_lambda) * long_memory + self.long_memory_lambda * track.output_embed

                is_pos_reshaped = is_pos.reshape((is_pos.shape[0], 1))
                updated_track.ref_pts = updated_ref_pts
                updated_track.long_memory = track.long_memory * ~is_pos_reshaped + long_memory * is_pos_reshaped
                updated_track.last_output = track.last_output * ~is_pos_reshaped + output_embed * is_pos_reshaped
                updated_query_embed = track.query_embed.clone()
                # Keep the masked assignment path even when `is_pos` is all False so
                # QueryUpdater stays connected to the autograd/DDP graph like the old implementation.
                updated_query_embed[is_pos] = query_feat[is_pos]
                updated_track.query_embed = updated_query_embed
                updated_tracks.append(updated_track)
            else:
                updated_tracks.append(track.clone())

        return updated_tracks

def build(config: dict):
    sig = inspect.signature(QueryUpdater)
    _cfg = {k: v for k, v in config["Query_updater"].items() if k in sig.parameters}
    return QueryUpdater(**_cfg)
