# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Positional encoding module."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


class PositionEmbeddingSine(nn.Module):
    """This is a more standard version of the position embedding."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ):
        """Initialize the PositionEmbeddingSine module.

        Args:
            num_pos_feats (int): Number of positional features.
            temperature (int): Temperature scaling factor.
            normalize (bool): Flag indicating whether to normalize the position embeddings.
            scale (Optional[float]): Scaling factor for the position embeddings. If None, default value is used.
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            msg = "normalize should be True if scale is passed"
            raise ValueError(msg)
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: torch.Tensor) -> torch.Tensor:
        """Forward function for PositionEmbeddingSine module."""
        if isinstance(tensor_list, torch.Tensor):
            x = tensor_list
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=x.device).type_as(x)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def gen_sineembed_for_position(pos_tensor: torch.Tensor) -> torch.Tensor:
    """Generate sine embeddings for position tensor.

    Args:
        pos_tensor (Tensor): Position tensor of shape (n_query, bs, num_dims).

    Returns:
        Tensor: Sine embeddings for position tensor of shape (n_query, bs, embedding_dim).
    """
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.int64, device=pos_tensor.device).type_as(pos_tensor)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    elif pos_tensor.size(-1) == 6:
        for i in range(2, 6):  # Compute sine embeds for l, r, t, b
            embed = pos_tensor[:, :, i] * scale
            pos_embed = embed[:, :, None] / dim_t
            pos_embed = torch.stack((pos_embed[:, :, 0::2].sin(), pos_embed[:, :, 1::2].cos()), dim=3).flatten(2)
            pos = pos_embed if i == 2 else torch.cat((pos, pos_embed), dim=2)
        pos = torch.cat((pos_y, pos_x, pos), dim=2)
    else:
        msg = f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}"
        raise ValueError(msg)
    return pos


class RopePositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for Vision Transformers.

    Computes sinusoidal position embeddings that are applied via rotation
    to query and key vectors in attention layers.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        base: Base frequency for computing periods.
        min_period: Minimum period (alternative to base).
        max_period: Maximum period (alternative to base).
        normalize_coords: How to normalize coordinates ('min', 'max', 'separate').
        shift_coords: Optional shift to apply to coordinates.
        jitter_coords: Optional jitter range for data augmentation.
        rescale_coords: Optional rescaling factor for coordinates.
        dtype: Data type for embeddings.
        device: Device for embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if embed_dim % (4 * num_heads) != 0:
            msg = f"embed_dim ({embed_dim}) must be divisible by 4 * num_heads ({4 * num_heads})"
            raise ValueError(msg)
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            msg = "Either `base` or `min_period`+`max_period` must be provided."
            raise ValueError(msg)

        d_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = d_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(d_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, h: int, w: int) -> tuple[Tensor, Tensor]:
        """Compute sin and cos position embeddings.

        Args:
            H: Height of the feature map.
            W: Width of the feature map.

        Returns:
            Tuple of (sin, cos) tensors for rotary position embedding.
        """
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_hw = max(h, w)
            coords_h = torch.arange(0.5, h, **dd) / max_hw  # [H]
            coords_w = torch.arange(0.5, w, **dd) / max_hw  # [W]
        elif self.normalize_coords == "min":
            min_hw = min(h, w)
            coords_h = torch.arange(0.5, h, **dd) / min_hw  # [H]
            coords_w = torch.arange(0.5, w, **dd) / min_hw  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, h, **dd) / h  # [h]
            coords_w = torch.arange(0.5, w, **dd) / w  # [W]
        else:
            msg = f"Unknown normalize_coords: {self.normalize_coords}"
            raise ValueError(msg)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            # min_period and max_period are guaranteed to be set when base is None
            if self.min_period is None or self.max_period is None:
                msg = "min_period and max_period must be set when base is None"
                raise RuntimeError(msg)
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods
