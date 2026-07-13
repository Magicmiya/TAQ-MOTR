# Modified from RT-DETRv2 (https://github.com/zheli-hub/RT-DETRv2)

import torch.nn.functional as F
import torch
from torch import Tensor, nn
from typing import Sequence

from ..ops.modules import MSDeformAttn


def _normalize_num_points(num_points: int | Sequence[int], num_levels: int) -> list[int]:
    if isinstance(num_points, Sequence) and not isinstance(num_points, (str, bytes)):
        points = [int(p) for p in num_points]
        if len(points) == 0:
            raise ValueError("num_points list cannot be empty.")
        if len(points) < num_levels:
            points.extend([points[-1]] * (num_levels - len(points)))
        elif len(points) > num_levels:
            raise ValueError(f"num_points length {len(points)} exceeds num_levels {num_levels}.")
        return points
    return [int(num_points)] * num_levels


def _spatial_shapes_to_list(value_spatial_shapes: Tensor | Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    if isinstance(value_spatial_shapes, torch.Tensor):
        shapes = value_spatial_shapes.detach().cpu().tolist()
    else:
        shapes = value_spatial_shapes
    return [(int(h), int(w)) for h, w in shapes]


def _validate_ref_levels(reference_points: Tensor, num_levels: int) -> int:
    n_ref_levels = reference_points.shape[2]
    if n_ref_levels not in (1, num_levels):
        raise ValueError(f"reference_points level dim must be 1 or {num_levels}, got {n_ref_levels}")
    return n_ref_levels


def _expand_reference_by_levels(reference_points: Tensor, repeats: Tensor, total_points: int) -> tuple[Tensor, Tensor]:
    n_ref_levels = reference_points.shape[2]
    if n_ref_levels == 1:
        ref_xy = reference_points[..., :2].expand(-1, -1, total_points, -1)
        ref_wh = reference_points[..., 2:].expand(-1, -1, total_points, -1)
    else:
        ref_xy = torch.repeat_interleave(reference_points[..., :2], repeats, dim=2)
        ref_wh = torch.repeat_interleave(reference_points[..., 2:], repeats, dim=2)
    return ref_xy, ref_wh


def _deformable_attention_core(
    value: Tensor,
    value_spatial_shapes: Tensor | Sequence[Sequence[int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
    num_points_list: list[int],
    method: str,
) -> Tensor:
    bs, _, n_head, c = value.shape
    _, len_q, _, _, _ = sampling_locations.shape

    spatial_shapes = _spatial_shapes_to_list(value_spatial_shapes)
    split_shape = [h * w for h, w in spatial_shapes]
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)

    sampling_grids = sampling_locations if method == "default" else (2 * sampling_locations - 1)
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(spatial_shapes):
        value_l = value_list[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l = sampling_locations_list[level]
        if method == "discrete":
            sampling_coord = (sampling_grid_l * torch.tensor([[w, h]], device=value.device) + 0.5).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, len_q * num_points_list[level], 2)
            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, len_q, num_points_list[level])
        else:
            sampling_value_l = F.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, len_q)
    return output.permute(0, 2, 1)


class MultiheadSelfAttention(nn.MultiheadAttention):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        x: Tensor,
        pos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        q = k = x if pos is None else x + pos
        out, weights = super().forward(
            q,
            k,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        return out, weights if need_weights else None


class CrossAttention(nn.Module):
    """
    Cross-attention using scaled_dot_product_attention.

    Mask convention:
    - key_padding_mask: bool [B, Nk], True means this key is ignored.
    - attn_mask:
      - bool mask where True means blocked,
      - or float additive mask.
      Shape: [Nq, Nk] or [B, Nq, Nk].
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        if self.internal_dim % num_heads != 0:
            raise ValueError("num_heads must divide internal_dim.")

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout_p = dropout
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def _separate_heads(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, self.num_heads, c // self.num_heads)
        return x.transpose(1, 2)  # [B, H, N, Dh]

    @staticmethod
    def _recombine_heads(x: Tensor) -> Tensor:
        bsz, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(bsz, n_tokens, n_heads * c_per_head)  # [B, N, C]

    @staticmethod
    def _to_additive_mask(mask: Tensor, dtype: torch.dtype) -> Tensor:
        if mask.dtype == torch.bool:
            neg = torch.finfo(dtype).min
            out = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
            return out.masked_fill(mask, neg)
        return mask.to(dtype=dtype)

    def _build_sdpa_mask(
        self,
        q: Tensor,
        k: Tensor,
        key_padding_mask: Tensor | None,
        attn_mask: Tensor | None = None,
    ) -> Tensor | None:
        bsz = q.shape[0]
        nk = k.shape[-2]
        dtype = q.dtype
        final_mask = None

        if attn_mask is not None:
            if attn_mask.dim() == 2:  # [Nq, Nk]
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.dim() == 3:  # [B, Nq, Nk]
                attn_mask = attn_mask[:, None, :, :]
            else:
                raise ValueError(f"attn_mask shape not supported: {tuple(attn_mask.shape)}")
            final_mask = self._to_additive_mask(attn_mask, dtype=dtype)

        if key_padding_mask is not None:
            expected_shape = (bsz, nk)
            if key_padding_mask.shape != expected_shape:
                raise ValueError(
                    f"key_padding_mask shape mismatch: expected {expected_shape}, got {tuple(key_padding_mask.shape)}"
                )
            pad_mask = key_padding_mask[:, None, None, :]  # [B, 1, 1, Nk]
            pad_additive = self._to_additive_mask(pad_mask, dtype=dtype)
            final_mask = pad_additive if final_mask is None else (final_mask + pad_additive)

        return final_mask

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q)
        k = self._separate_heads(k)
        v = self._separate_heads(v)

        sdpa_mask = self._build_sdpa_mask(q=q, k=k, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        # If all keys are masked for a sample, define its output as zero.
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=1)
            if torch.any(all_masked):
                out = out.clone()
                out[all_masked] = 0.0

        if not torch.isfinite(out).all():
            raise FloatingPointError("CrossAttention output contains NaN/Inf.")
        return out


class MSDeformableAttentionTorch(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int | Sequence[int] = 4,
        method: str = "default",
        offset_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale
        self.method = method

        self.num_points_list = _normalize_num_points(num_points, num_levels)
        num_points_scale = [1 / n for n in self.num_points_list for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(self.num_points_list)
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._last_visual_payload = None
        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * torch.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def pop_last_visual_payload(self):
        payload = self._last_visual_payload
        self._last_visual_payload = None
        return payload

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        value_spatial_shapes: Tensor | Sequence[Sequence[int]],
        value_mask: Tensor | None = None,
    ) -> Tensor:
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], 0.0)
        value = value.reshape(bs, len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(bs, len_q, self.num_heads, sum(self.num_points_list), 2)
        attention_weights = self.attention_weights(query).reshape(bs, len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        total_points = sum(self.num_points_list)
        repeats = torch.tensor(self.num_points_list, device=query.device, dtype=torch.long)
        
        if reference_points.shape[-1] == 2:
            n_ref_levels = _validate_ref_levels(reference_points=reference_points, num_levels=self.num_levels)
            spatial_shapes = torch.as_tensor(
                _spatial_shapes_to_list(value_spatial_shapes), dtype=query.dtype, device=query.device
            )
            offset_normalizer = torch.repeat_interleave(spatial_shapes.flip(-1), repeats, dim=0)
            if n_ref_levels == 1:
                ref_xy = reference_points.expand(-1, -1, total_points, -1)
            else:
                ref_xy = torch.repeat_interleave(reference_points, repeats, dim=2)
            sampling_locations = ref_xy[:, :, None] + sampling_offsets / offset_normalizer[None, None, None]
        elif reference_points.shape[-1] == 4:
            _validate_ref_levels(reference_points=reference_points, num_levels=self.num_levels)
            num_points_scale = self.num_points_scale.to(device=query.device, dtype=query.dtype).reshape(1, 1, 1, -1, 1)
            ref_xy, ref_wh = _expand_reference_by_levels(
                reference_points=reference_points,
                repeats=repeats,
                total_points=total_points,
            )
            sampling_locations = (
                ref_xy[:, :, None] + sampling_offsets * num_points_scale * ref_wh[:, :, None] * self.offset_scale
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead."
            )

        # Mirror the CUDA attention module so decoder-side visualization logic can stay backend-agnostic.
        self._last_visual_payload = {
            "sampling_locations": sampling_locations,
            "attention_weights": attention_weights,
            "value_spatial_shapes": value_spatial_shapes,
            "reference_points": reference_points,
        }
        output = _deformable_attention_core(
            value=value,
            value_spatial_shapes=value_spatial_shapes,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            num_points_list=self.num_points_list,
            method=self.method,
        )
        return self.output_proj(output)


def build_ms_deformable_attention(
    embed_dim: int = 256,
    num_heads: int = 8,
    num_levels: int = 4,
    num_points: int | Sequence[int] = 4,
    method: str = "CUDA",
    offset_scale: float = 0.5,
    sigmoid_attn: bool = False,
    visualize: bool = False,
) -> nn.Module:
    num_points_list = _normalize_num_points(num_points, num_levels)
    if method == "CUDA":
        if len(set(num_points_list)) != 1:
            raise ValueError("MSDeformAttn CUDA backend requires the same num_points for each level.")
        return MSDeformAttn(
            embed_dim=embed_dim,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points_list[0],
            offset_scale=offset_scale,
            sigmoid_attn=sigmoid_attn,
            visualize=visualize,
        )
    return MSDeformableAttentionTorch(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points_list,
        method=method,
        offset_scale=offset_scale,
    )
