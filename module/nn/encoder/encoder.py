# @Author       : Ruopeng Gao
# @Date         : 2025/7/3
# @Description  : Used for constructing multi-scale features and their position embeddings
# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)
import copy
import inspect

import torch
import torch.nn as nn

from torch import Tensor
from typing import List

from module.nn.common.position_embedding import build as build_pose_emb
from module.nn.common import NestedTensor
from module.nn.common import freeze_batch_norm2d, no_local_batch_norm2d
from module.nn.common.transformer import build_ms_deformable_attention


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def get_activation_layer(activation: str) -> nn.Module:
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    if activation == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {activation}")


class DeformableEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = get_clones(module=encoder_layer, n=num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes: Tensor, valid_ratios: Tensor, device: torch.device) -> Tensor:
        reference_points_list = []
        for lvl, (h_i, w_i) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h_i - 0.5, int(h_i), dtype=torch.float32, device=device),
                torch.linspace(0.5, w_i - 0.5, int(w_i), dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h_i)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w_i)
            ref = torch.stack((ref_x, ref_y), dim=-1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, dim=1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        pos: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        attn_method: str = "CUDA",
    ):
        super().__init__()
        self.self_attn = build_ms_deformable_attention(
            embed_dim=d_model,
            num_heads=n_heads,
            num_levels=n_levels,
            num_points=n_points,
            method=attn_method,
            sigmoid_attn=False,
            visualize=False,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_layer(activation=activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: Tensor) -> Tensor:
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src: Tensor,
        pos: Tensor | None,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        # TAQ_MOTR MSDeformAttn computes level_start_index internally, so keep the MeMOTR-like layer
        # signature for structure alignment but call the local operator with its native argument layout.
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class MeMOTR_Encoder(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        feat_strides: list[int],
        feature_levels: int,
        hidden_dim: int = 256,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        nhead: int = 8,
        num_points: int = 4,
        attn_method: str = "CUDA",
    ):
        super().__init__()
        if len(in_channels) != feature_levels:
            raise ValueError(f"feature_levels={feature_levels} must match len(in_channels)={len(in_channels)}")
        if len(feat_strides) != feature_levels:
            raise ValueError(f"feature_levels={feature_levels} must match len(feat_strides)={len(feat_strides)}")

        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.feature_levels = feature_levels
        self.hidden_dim = hidden_dim
        self.position_embedding = build_pose_emb({"HIDDEN_DIM": hidden_dim})

        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=channel, out_channels=hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
                for channel in in_channels
            ]
        )

        encoder_layer = DeformableEncoderLayer(
            d_model=hidden_dim,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=feature_levels,
            n_heads=nhead,
            n_points=num_points,
            attn_method=attn_method,
        )
        self.encoder = DeformableEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1.0)
            nn.init.constant_(proj[0].bias, 0.0)

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        valid_h = torch.sum(~mask[:, :, 0], dim=1)
        valid_w = torch.sum(~mask[:, 0, :], dim=1)
        ratio_h = valid_h.float() / mask.shape[1]
        ratio_w = valid_w.float() / mask.shape[2]
        return torch.stack([ratio_w, ratio_h], dim=-1)

    def forward(self, features: List[NestedTensor], img_masks: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if len(features) != self.feature_levels:
            raise ValueError(f"Expected {self.feature_levels} feature levels, got {len(features)}")

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        level_masks = []

        for level, feature in enumerate(features):
            src, mask = feature.decompose()
            if mask is None:
                if img_masks is None:
                    raise ValueError("img_masks is required when feature masks are missing")
                # Keep encoder input compatible with TAQ_MOTR call sites by rebuilding per-level masks from image mask.
                mask = torch.nn.functional.interpolate(
                    img_masks[None].float(),
                    size=src.shape[-2:],
                )[0].to(torch.bool)

            proj_src = self.input_proj[level](src)
            pos_embed = self.position_embedding(NestedTensor(proj_src, mask))

            _, _, height, width = proj_src.shape
            spatial_shapes.append((height, width))
            level_masks.append(mask)
            src_flatten.append(proj_src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            lvl_pos_embed_flatten.append(pos_embed.flatten(2).transpose(1, 2))

        src_flatten_tensor = torch.cat(src_flatten, dim=1)
        mask_flatten_tensor = torch.cat(mask_flatten, dim=1)
        pos_flatten_tensor = torch.cat(lvl_pos_embed_flatten, dim=1)
        spatial_shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten_tensor.device)
        level_start_index = torch.cat(
            [
                spatial_shapes_tensor.new_zeros((1,)),
                spatial_shapes_tensor.prod(1).cumsum(0)[:-1],
            ]
        )
        valid_ratios = torch.stack([self.get_valid_ratio(mask) for mask in level_masks], dim=1)

        memory = self.encoder(
            src=src_flatten_tensor,
            spatial_shapes=spatial_shapes_tensor,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            pos=pos_flatten_tensor,
            padding_mask=mask_flatten_tensor,
        )
        return memory, mask_flatten_tensor


def build(config):
    encoder_cfg = config["Encoder"]
    module_cfg = {}
    sig = inspect.signature(MeMOTR_Encoder)
    for key, value in encoder_cfg.items():
        if key in sig.parameters:
            module_cfg[key] = value

    if "enc_act" in encoder_cfg and "activation" in sig.parameters:
        module_cfg["activation"] = encoder_cfg["enc_act"]
    if "num_points" in encoder_cfg:
        num_points = encoder_cfg["num_points"]
        module_cfg["num_points"] = num_points[0] if isinstance(num_points, list) else num_points

    model = MeMOTR_Encoder(**module_cfg)
    batch_norm = encoder_cfg.get("batch_norm", "normal")
    if batch_norm == "freeze":
        model = freeze_batch_norm2d(model)
    elif batch_norm == "no_local":
        model = no_local_batch_norm2d(model)
    return model
