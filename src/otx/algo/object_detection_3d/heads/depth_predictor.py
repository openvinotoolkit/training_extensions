# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""depth predictor transformer head for 3d object detection."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn import functional

from otx.algo.detection.necks.hybrid_encoder import TransformerEncoder, TransformerEncoderLayer


class DepthPredictor(nn.Module):
    """Depth predictor and depth encoder."""

    def __init__(self, depth_num_bins: int, depth_min: float, depth_max: float, hidden_dim: int) -> None:
        """Initialize depth predictor and depth encoder.

        Args:
            depth_num_bins (int): The number of depth bins.
            depth_min (float): The minimum depth value.
            depth_max (float): The maximum depth value.
            hidden_dim (int): The dimension of the hidden layer.
        """
        super().__init__()
        self.depth_max = depth_max

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # Create modules
        d_model = hidden_dim
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model),
        )
        self.proj = nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size=(1, 1)), nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size=(1, 1)), nn.GroupNorm(32, d_model))

        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
        )

        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))

        depth_encoder_layer = TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=256, dropout=0.1, normalize_before=True)

        self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)

        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)

    def forward(
        self,
        feature: list[torch.Tensor],
        mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the DepthPredictor.

        Args:
            feature (List[torch.Tensor]): The list of input feature tensors.
            mask (torch.Tensor): The mask tensor.
            pos (torch.Tensor): The positional tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The output tensors.
                - depth_logits: The depth logits tensor.
                - depth_embed: The depth embedding tensor.
                - weighted_depth: The weighted depth tensor.
                - depth_pos_embed_ip: The interpolated depth positional embedding tensor.
        """
        # foreground depth map
        src_16 = self.proj(feature[1])
        src_32 = self.upsample(functional.interpolate(feature[2], size=src_16.shape[-2:], mode="bilinear"))
        src_8 = self.downsample(feature[0])
        src = (src_8 + src_16 + src_32) / 3

        src = self.depth_head(src)
        depth_logits = self.depth_classifier(src)

        depth_probs = functional.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)
        # depth embeddings with depth positional encodings
        b, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)

        depth_embed = self.depth_encoder(src, mask, pos)
        depth_embed = depth_embed.permute(1, 2, 0).reshape(b, c, h, w)
        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        depth_embed = depth_embed + depth_pos_embed_ip

        return depth_logits, depth_embed, weighted_depth, depth_pos_embed_ip

    def interpolate_depth_embed(self, depth: torch.Tensor) -> torch.Tensor:
        """Interpolate depth embeddings based on depth values.

        Args:
            depth (torch.Tensor): The depth tensor.

        Returns:
            torch.Tensor: The interpolated depth embeddings.
        """
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        return pos.permute(0, 3, 1, 2)

    def interpolate_1d(self, coord: torch.Tensor, embed: nn.Embedding) -> torch.Tensor:
        """Interpolate 1D embeddings based on coordinates.

        Args:
            coord (torch.Tensor): The coordinate tensor.
            embed (nn.Embedding): The embedding module.

        Returns:
            torch.Tensor: The interpolated embeddings.
        """
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
