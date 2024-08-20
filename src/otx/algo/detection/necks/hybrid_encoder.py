# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Hybrid Encoder module for detection task, modified from https://github.com/lyuwenyu/RT-DETR."""

from __future__ import annotations

import copy
from functools import partial
from typing import Callable

import torch
from torch import nn

from otx.algo.detection.layers import CSPRepLayer
from otx.algo.modules import Conv2dModule
from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.norm import build_norm_layer

__all__ = ["HybridEncoder"]


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.GELU,
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos_embed: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class HybridEncoder(BaseModule):
    """HybridEncoder for RTDetr.

    Args:
        in_channels (list[int], optional): List of input channels for each feature map.
            Defaults to [512, 1024, 2048].
        feat_strides (list[int], optional): List of stride values for
            each feature map. Defaults to [8, 16, 32].
        hidden_dim (int, optional): Hidden dimension size. Defaults to 256.
        nhead (int, optional): Number of attention heads in the transformer encoder.
                Defaults to 8.
        dim_feedforward (int, optional): Dimension of the feedforward network
            in the transformer encoder. Defaults to 1024.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        enc_activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.GELU``.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(build_norm_layer, nn.BatchNorm2d, layer_name="norm")``.
        use_encoder_idx (list[int], optional): List of indices of the encoder to use.
            Defaults to [2].
        num_encoder_layers (int, optional): Number of layers in the transformer encoder.
            Defaults to 1.
        pe_temperature (float, optional): Temperature parameter for positional encoding.
            Defaults to 10000.
        expansion (float, optional): Expansion factor for the CSPRepLayer.
            Defaults to 1.0.
        depth_mult (float, optional): Depth multiplier for the CSPRepLayer.
            Defaults to 1.0.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.SiLU``.
        eval_spatial_size (tuple[int, int] | None, optional): Spatial size for
            evaluation. Defaults to None.
    """

    def __init__(
        self,
        in_channels: list[int] = [512, 1024, 2048],  # noqa: B006
        feat_strides: list[int] = [8, 16, 32],  # noqa: B006
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        enc_activation: Callable[..., nn.Module] = nn.GELU,
        normalization: Callable[..., nn.Module] = partial(build_norm_layer, nn.BatchNorm2d, layer_name="norm"),
        use_encoder_idx: list[int] = [2],  # noqa: B006
        num_encoder_layers: int = 1,
        pe_temperature: float = 10000,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        activation: Callable[..., nn.Module] = nn.SiLU,
        eval_spatial_size: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the HybridEncoder module."""
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                ),
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_activation,
        )

        self.encoder = nn.ModuleList(
            [TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))],
        )

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                Conv2dModule(
                    hidden_dim,
                    hidden_dim,
                    1,
                    1,
                    normalization=build_norm_layer(normalization, num_features=hidden_dim),
                    activation=build_activation_layer(activation),
                ),
            )
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    activation=activation,
                    expansion=expansion,
                    normalization=normalization,
                ),
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                Conv2dModule(
                    hidden_dim,
                    hidden_dim,
                    3,
                    2,
                    padding=1,
                    normalization=build_norm_layer(normalization, num_features=hidden_dim),
                    activation=build_activation_layer(activation),
                ),
            )
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    activation=activation,
                    expansion=expansion,
                    normalization=normalization,
                ),
            )

    def init_weights(self) -> None:
        """Initialize weights."""
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int,
        h: int,
        embed_dim: int = 256,
        temperature: float = 10000.0,
    ) -> torch.Tensor:
        """Build 2D sin-cos position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            msg = "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
            raise ValueError(msg)
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass."""
        if len(feats) != len(self.in_channels):
            msg = f"Input feature size {len(feats)} does not match the number of input channels {len(self.in_channels)}"
            raise ValueError(msg)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(
                        src_flatten.device,
                    )
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None)
                    if pos_embed is not None:
                        pos_embed = pos_embed.to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = nn.functional.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs
