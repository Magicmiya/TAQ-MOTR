# Modified from MeMOTR (https://github.com/MCG-NJU/MeMOTR)

import torch
import math
from torch import nn

from .nested_tensor import NestedTensor
from .spatiotemporal_embedding import SinePositionEmbedding2D, coords_from_mask


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        if self.normalize is True and self.scale is None:
            raise ValueError("Scale should be NOT NONE when normalize is True.")
        if self.scale is not None and self.normalize is False:
            raise ValueError("Normalize should be True when scale is not None.")
        self.encoder = SinePositionEmbedding2D(
            num_pos_feats=self.num_pos_feats,
            temperature=self.temperature,
            axis_order="yx",
        )

    def forward(self, ntensor: NestedTensor) -> torch.Tensor:
        _, masks = ntensor.decompose()
        assert masks is not None, "Masks in ntensor should be NOT NONE."
        coords = coords_from_mask(
            mask=masks,
            normalize=self.normalize,
            scale=1.0 if self.scale is None else self.scale,
            flatten=False,
        )
        pos_embed = self.encoder(coords).permute(0, 3, 1, 2)
        return pos_embed


def build(config: dict):
    assert config["HIDDEN_DIM"] % 2 == 0, f"Hidden dim should be 2x, but get {config['HIDDEN_DIM']}."
    num_pos_feats = config["HIDDEN_DIM"] / 2
    return PositionEmbeddingSine(num_pos_feats=num_pos_feats, normalize=True, scale=2*math.pi, temperature=20)
