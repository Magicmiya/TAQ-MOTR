# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About: 在 MOTR 的对应文件中增加了部分注释，助于理解。
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)
# Modified from RT-DETRv2 (https://github.com/zheli-hub/RT-DETRv2)


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from typing import List

from ..functions import MSDeformAttnFunction
from utils.visualizer import TensorHook


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels: int | list[int] = 4,
        num_points=4,
        offset_scale=0.5,
        sigmoid_attn=False,
        visualize=False,
    ):
        """
        Multi-Scale Deformable Attention Module
        :param embed_dim      hidden dimension
        :param num_levels     number of feature levels
        :param num_heads      number of attention heads
        :param num_points     number of sampling points per attention head per feature level
        :param sigmoid_attn 使用 sigmoid 代替 softmax 计算 attention score，在原本的 Deformable DETR 中没有。
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads, but got {} and {}'.format(embed_dim, num_heads))
        _d_per_head = embed_dim // num_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set embed_dim in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )
        self.im2col_step = 64
        self.sigmoid_attn = sigmoid_attn

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.head_dim = embed_dim // num_heads
        self.offset_scale = offset_scale

        # self.n_points = n_points
        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.visualize = visualize
        self._last_visual_payload = None

        self.reset_parameters()

    def pop_last_visual_payload(self):
        payload = self._last_visual_payload
        self._last_visual_payload = None
        return payload

    def reset_parameters(self):
        # sampling_offsets
        constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        constant_(self.attention_weights.weight, 0)
        constant_(self.attention_weights.bias, 0)

        # proj
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias, 0)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias.data, 0)

    @TensorHook(["attention_weights", "sampling_locations", "value_spatial_shapes"])
    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_mask: torch.Tensor = None,
    ):
        """
        Args:
            query(Tensor)               (bs, query_length, C)
            reference_points(Tensor)    (bs, query_length, n_levels, 2), range in [0, 1], top-left (0,0),
                                            bottom-right (1, 1), including padding area
                                        (bs, query_length, n_levels, 4), add additional (w, h) form reference boxes
            value(Tensor)               (bs, value_length, C)
            value_spatial_shapes(List)  [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask(Tensor)          [bs, value_length], True for padding elements, False for non-padding elements

        Returns:
            output                      [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)
        h_w_products = value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]  # [n_levels]
        level_start_index = torch.nn.functional.pad(torch.cumsum(h_w_products[:-1], dim=0), (1, 0), value=0)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(
            bs, Len_q, self.num_heads, len(self.num_points_list), self.num_points_list[0], 2
        )

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        if self.sigmoid_attn:
            attention_weights = attention_weights.sigmoid().reshape(
                bs, Len_q, self.num_heads, sum(self.num_points_list)
            )
        else:
            attention_weights = F.softmax(attention_weights, dim=-1).reshape(
                bs, Len_q, self.num_heads, sum(self.num_points_list)
            )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4](x,y,w,h)
            # sampling_offsets [8, 480, 8,    12, 2]()
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).clone().reshape(self.num_levels, -1, 1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, None, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, None, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )
        """
        value(tensor)                   : (batch_size, len_val, num_heads, head_dim）
        value_spatial_shapes(tensor)   : (n_levels, 2)
        level_start_index(tensor)      : (n_levels)
        sampling_locations(tensor)      : (batch_size,len_query, n_heads, n_levels, n_points, 2)
        attention_weights(tensor)       : (batch_size,len_query, n_heads, n_levels, n_points)
        """
        # Here is a bug: MSDeformAttnFunction only accept the same number of sample-point used in each layer.
        # Sine we used the same number of sample-point(4 for each layer) we directly reshape the loc & weight.
        # ^_^/ That's right, I'm slacking off here，if you need you can padding the sample step
        sampling_locations = sampling_locations.reshape(
            bs, Len_q, self.num_heads, len(self.num_points_list), self.num_points_list[0], 2
        )
        attention_weights = attention_weights.reshape(
            bs, Len_q, self.num_heads, len(self.num_points_list), self.num_points_list[0]
        )
        # Keep the last cross-attention sampling payload on-module so decoder code can emit a semantically named event.
        self._last_visual_payload = {
            "sampling_locations": sampling_locations,
            "attention_weights": attention_weights,
            "value_spatial_shapes": value_spatial_shapes,
            "reference_points": reference_points,
        }
        output = MSDeformAttnFunction.apply(
            value, value_spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step
        )
        output = self.output_proj(output)
        return output
