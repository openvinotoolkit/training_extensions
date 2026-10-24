# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EdgeCrafter instance-segmentation head (query-conditioned dot-product mask head).

Used by :class:`~getitune.backend.lightning.models.detection.heads.ec_decoder.ECTransformer`
when ``mask_downsample_ratio`` is set (i.e. for :class:`EdgeCrafterInst`). Kept as its own
module in ``instance_segmentation/heads/`` (rather than inlined in ``ec_decoder.py``) to
keep the primary decoder file focused on detection logic and to group this
instance-segmentation-specific head alongside its siblings (``fcn_mask_head.py``,
``rtmdet_inst_head.py``, etc.), following the "one head per file" convention used
throughout ``instance_segmentation/heads/``.

Modified from EdgeCrafter (https://github.com/Intellindust-AI-Lab/EdgeCrafter).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

__all__ = ["SegmentationHead"]


class _DepthwiseConvBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        res = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(self.pwconv1(self.norm(x)))
        return x.permute(0, 3, 1, 2) + res


class _MLPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        # Named "layers" to match checkpoint key layout (layers.0, layers.2).
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return x + self.layers(self.norm_in(x))


class SegmentationHead(nn.Module):
    """Lightweight dot-product segmentation head.

    Args:
        in_dim: Feature dimension.
        num_blocks: Number of DepthwiseConvBlocks applied to spatial features.
        downsample_ratio: Spatial downsampling ratio for the mask output.
        image_size: Reference image size ``(H, W)`` (used to compute mask target size).
    """

    def __init__(
        self,
        in_dim: int,
        num_blocks: int,
        downsample_ratio: int = 4,
        image_size: tuple[int, int] | list[int] = (640, 640),
    ) -> None:
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.image_size = tuple(image_size)
        self.blocks = nn.ModuleList([_DepthwiseConvBlock(in_dim) for _ in range(num_blocks)])
        # 1x1 conv projects spatial features channel-wise (matches checkpoint key layout).
        self.spatial_features_proj = nn.Conv2d(in_dim, in_dim, 1)
        self.query_features_block = _MLPBlock(in_dim)
        # Linear projection on query features (matches checkpoint key layout).
        self.query_features_proj = nn.Linear(in_dim, in_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        spatial_features: Tensor,
        query_features: list[Tensor],
    ) -> list[Tensor]:
        """Forward pass during training (one mask per decoder layer)."""
        h_out = self.image_size[0] // self.downsample_ratio
        w_out = self.image_size[1] // self.downsample_ratio
        sf = F.interpolate(spatial_features, size=(h_out, w_out), mode="bilinear", align_corners=False)

        mask_logits = []
        for block, qf_in in zip(self.blocks, query_features):
            sf = block(sf)
            sf_proj = self.spatial_features_proj(sf)
            qf = self.query_features_proj(self.query_features_block(qf_in))
            mask_logits.append(torch.einsum("bchw,bnc->bnhw", sf_proj, qf) + self.bias)
        return mask_logits

    def forward_export(
        self,
        spatial_features: Tensor,
        query_features: list[Tensor],
    ) -> list[Tensor]:
        """Forward at export time (single query feature, no dropout)."""
        if len(query_features) != 1:
            msg = f"forward_export expects exactly one query feature tensor, got {len(query_features)}."
            raise ValueError(msg)
        h_out = self.image_size[0] // self.downsample_ratio
        w_out = self.image_size[1] // self.downsample_ratio
        sf = F.interpolate(spatial_features, size=(h_out, w_out), mode="bilinear", align_corners=False)
        for block in self.blocks:
            sf = block(sf)
        sf = self.spatial_features_proj(sf)
        qf = self.query_features_proj(self.query_features_block(query_features[0]))
        return [torch.einsum("bchw,bnc->bnhw", sf, qf) + self.bias]
