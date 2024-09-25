# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""depth predictor transformer head for 3d object detection."""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module = None) -> None:
        """Initialize the TransformerEncoder.

        Args:
            encoder_layer (nn.Module): The encoder layer module.
            num_layers (int): The number of encoder layers.
            norm (nn.Module, optional): The normalization module. Defaults to None.
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TransformerEncoder.

        Args:
            src (torch.Tensor): The source tensor.
            src_key_padding_mask (torch.Tensor): The mask for source key padding.
            pos (torch.Tensor): The positional encoding tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        output = src

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:
        """Initialize the TransformerEncoderLayer.

        Args:
            d_model (int): The dimension of the input feature.
            nhead (int): The number of attention heads.
            dim_feedforward (int, optional): The dimension of the feedforward network. Defaults to 2048.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            activation (Callable[..., nn.Module], optional): The activation function. Defaults to nn.ReLU.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = LinearAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()

    def _with_pos_embed(self, tensor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the TransformerEncoderLayer.

        Args:
            src (torch.Tensor): The source tensor.
            src_key_padding_mask (torch.Tensor): The mask for source key padding.
            pos (torch.Tensor): The positional encoding tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The output tensors.
                - depth_logits: The depth logits tensor.
                - depth_embed: The depth embedding tensor.
                - weighted_depth: The weighted depth tensor.
                - depth_pos_embed_ip: The interpolated depth positional embedding tensor.
        """
        q = k = self._with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DepthPredictor(nn.Module):
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

        depth_encoder_layer = TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=256, dropout=0.1)

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
        src_32 = self.upsample(F.interpolate(feature[2], size=src_16.shape[-2:], mode="bilinear"))
        src_8 = self.downsample(feature[0])
        src = (src_8 + src_16 + src_32) / 3

        src = self.depth_head(src)
        depth_logits = self.depth_classifier(src)

        depth_probs = F.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)
        # depth embeddings with depth positional encodings
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)

        depth_embed = self.depth_encoder(src, mask, pos)
        depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)
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
        pos = pos.permute(0, 3, 1, 2)
        return pos

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
