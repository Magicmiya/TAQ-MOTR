import math

import torch
import torch.nn as nn


def coords_from_mask(
    mask: torch.Tensor,
    normalize: bool = True,
    scale: float = 2 * math.pi,
    eps: float = 1e-6,
    flatten: bool = False,
) -> torch.Tensor:
    """
    Build XY coordinates from padding mask.

    Args:
        mask: [B, H, W], True means padding.
        normalize: normalize coordinates to [0, scale] by valid region.
        scale: coordinate scale used for sine encoding.
        flatten: return [B, H*W, 2] when True, else [B, H, W, 2].
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be [B, H, W], got {tuple(mask.shape)}")

    not_mask = ~mask
    y = not_mask.cumsum(dim=1, dtype=torch.float32)
    x = not_mask.cumsum(dim=2, dtype=torch.float32)
    if normalize:
        y = (y - 0.5) / (y[:, -1:, :] + eps) * scale
        x = (x - 0.5) / (x[:, :, -1:] + eps) * scale
    coords = torch.stack([x, y], dim=-1)
    if flatten:
        coords = coords.flatten(1, 2)
    return coords


def coords_from_hw(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    scale: float = 1.0,
    flatten: bool = True,
    w_first: bool = False,
) -> torch.Tensor:
    """
    Build XY mesh-grid coordinates.

    Args:
        w_first:
            - False: standard [H, W] layout.
            - True: legacy [W, H] flatten order (for compatibility).
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be > 0, got {(height, width)}")

    if w_first:
        grid_w = torch.arange(width, dtype=dtype, device=device)
        grid_h = torch.arange(height, dtype=dtype, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        coords = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(0) * scale
    else:
        grid_y = torch.arange(height, dtype=dtype, device=device)
        grid_x = torch.arange(width, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0) * scale

    if flatten:
        coords = coords.flatten(1, 2)
    return coords


def _sine_position_embedding(
    coords_xy: torch.Tensor,
    num_pos_feats: int,
    temperature: float = 10000.0,
    axis_order: str = "yx",
) -> torch.Tensor:
    """
    Sine-cosine positional embedding from explicit coordinates.

    Args:
        coords_xy: [..., 2] where last dim is (x, y).
        num_pos_feats: feature dim per axis.
        axis_order: "yx" -> [y_embed, x_embed], "xy" -> [x_embed, y_embed].
    """
    if coords_xy.shape[-1] != 2:
        raise ValueError(f"coords_xy last dim must be 2, got {coords_xy.shape[-1]}")
    if num_pos_feats <= 0:
        raise ValueError(f"num_pos_feats must be > 0, got {num_pos_feats}")

    x = coords_xy[..., 0]
    y = coords_xy[..., 1]
    dim_i = torch.arange(num_pos_feats, dtype=torch.float32, device=coords_xy.device)
    dim_i = temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / num_pos_feats)

    x_embed = x[..., None] / dim_i
    y_embed = y[..., None] / dim_i
    x_embed = torch.stack((x_embed[..., 0::2].sin(), x_embed[..., 1::2].cos()), dim=-1).flatten(-2)
    y_embed = torch.stack((y_embed[..., 0::2].sin(), y_embed[..., 1::2].cos()), dim=-1).flatten(-2)

    if axis_order == "yx":
        return torch.cat((y_embed, x_embed), dim=-1)
    if axis_order == "xy":
        return torch.cat((x_embed, y_embed), dim=-1)
    raise ValueError(f"axis_order must be 'yx' or 'xy', got {axis_order}")


def box_sine_embedding(
    pos: torch.Tensor,
    num_pos_feats: int = 64,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
) -> torch.Tensor:
    """
    Sine-cosine embedding for box-like coordinates (..., D).
    Compatible with original pos_to_pos_embed behavior.
    """
    pos = pos * scale
    dim_i = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_i = temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / num_pos_feats)
    pos_embed = pos[..., None] / dim_i
    pos_embed = torch.stack((pos_embed[..., 0::2].sin(), pos_embed[..., 1::2].cos()), dim=-1)
    pos_embed = torch.flatten(pos_embed, start_dim=-3)
    return pos_embed


class SinePositionEmbedding2D(nn.Module):
    """
    Coordinate-only 2D sine position encoder.
    Grid generation should happen outside this module.
    """

    def __init__(self, num_pos_feats: int = 64, temperature: float = 10000.0, axis_order: str = "yx"):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.axis_order = axis_order

    def forward(self, coords_xy: torch.Tensor) -> torch.Tensor:
        return _sine_position_embedding(
            coords_xy=coords_xy,
            num_pos_feats=self.num_pos_feats,
            temperature=self.temperature,
            axis_order=self.axis_order,
        )


class RandomFourierEncoder(nn.Module):
    """
    Encode continuous inputs with random Fourier features and project to hidden_dim.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_frequencies: int = 64,
        scale: float = 1.0,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if num_frequencies <= 0:
            raise ValueError(f"num_frequencies must be > 0, got {num_frequencies}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies
        gaussian = scale * torch.randn(input_dim, num_frequencies)
        self.register_buffer("gaussian_matrix", gaussian)
        self.proj = nn.Linear(num_frequencies * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dim {self.input_dim}, got {x.shape[-1]}")
        x_proj = (2.0 * math.pi) * x @ self.gaussian_matrix
        fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.norm(self.proj(fourier))


class SpatioTemporalEmbedding(nn.Module):
    """Build tracker memory positional encoding as spatial + temporal."""

    def __init__(
        self,
        hidden_dim: int,
        num_temporal_bins: int = 32,
        spatial_encoder: nn.Module | None = None,
        num_frequencies: int = 64,
        scale: float = 1.0,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if num_temporal_bins <= 0:
            raise ValueError(f"num_temporal_bins must be > 0, got {num_temporal_bins}")

        self.hidden_dim = hidden_dim
        self.num_temporal_bins = num_temporal_bins
        self.spatial_encoder = spatial_encoder or RandomFourierEncoder(
            input_dim=4,
            hidden_dim=hidden_dim,
            num_frequencies=num_frequencies,
            scale=scale,
        )
        self.temporal_embed = nn.Embedding(num_temporal_bins, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.temporal_embed.weight, std=0.02)

    def forward(
        self,
        boxes: torch.Tensor,
        t_pos_idx: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            boxes: [B, Nt, 4], normalized to [0, 1].
            t_pos_idx: [B, Nt], temporal index (0 = nearest).
            valid_mask: [B, Nt], True for valid tokens (non-padding).
        """
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError(f"boxes must be [B, Nt, 4], got {tuple(boxes.shape)}")
        if t_pos_idx.shape != boxes.shape[:2]:
            raise ValueError(f"t_pos_idx shape {tuple(t_pos_idx.shape)} != boxes[:2] {tuple(boxes.shape[:2])}")
        if valid_mask is not None and valid_mask.shape != boxes.shape[:2]:
            raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} != boxes[:2] {tuple(boxes.shape[:2])}")

        spatial_pos = self.spatial_encoder(boxes.to(dtype=torch.float32))
        temporal_idx = t_pos_idx.to(device=boxes.device, dtype=torch.long).clamp_(0, self.num_temporal_bins - 1)
        temporal_pos = self.temporal_embed(temporal_idx)
        st_pos = spatial_pos + temporal_pos
        if valid_mask is not None:
            st_pos = st_pos * valid_mask.unsqueeze(-1).to(st_pos.dtype)
        return st_pos
