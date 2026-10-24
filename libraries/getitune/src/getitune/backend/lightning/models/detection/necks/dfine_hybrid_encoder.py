# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""D-FINE Hybrid Encoder.

Modified from D-FINE (https://github.com/Peterande/D-FINE).
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, ClassVar, Literal

import torch
import torch.nn.functional as f
from torch import Tensor, nn

from getitune.backend.lightning.models.common.layers.transformer_layers import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from getitune.backend.lightning.models.detection.layers.csp_layer import CSPRepLayer
from getitune.backend.lightning.models.detection.utils.utils import auto_pad
from getitune.backend.lightning.models.modules.activation import build_activation_layer
from getitune.backend.lightning.models.modules.conv_module import Conv2dModule
from getitune.backend.lightning.models.modules.norm import build_norm_layer

# =============================================================================
# Helper Layers
# =============================================================================


class ConvNormLayer(nn.Module):
    """Convolution + BatchNorm + Activation layer.

    Args:
        ch_in: Input channels.
        ch_out: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        groups: Number of groups for grouped convolution.
        padding: Padding size. If None, uses (kernel_size-1)//2.
        bias: Whether to use bias in convolution.
        act: Activation function name or None.
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        padding: int | None = None,
        bias: bool = False,
        act: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=groups,
            padding=padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else act()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.act(self.norm(self.conv(x)))


class ConvNormLayerFusable(nn.Module):
    """Fusable Convolution + BatchNorm + Activation layer.

    Supports fusing Conv and BatchNorm for deployment optimization.

    Args:
        ch_in: Input channels.
        ch_out: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        groups: Number of groups for grouped convolution.
        padding: Padding size. If None, uses (kernel_size-1)//2.
        bias: Whether to use bias in convolution.
        act: Activation function class or None.
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        padding: int | None = None,
        bias: bool = False,
        act: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=groups, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else act()
        # Store params for deployment conversion
        self._ch_in = ch_in
        self._ch_out = ch_out
        self._kernel_size = kernel_size
        self._stride = stride
        self._groups = groups
        self._padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if hasattr(self, "conv_bn_fused"):
            return self.act(self.conv_bn_fused(x))
        return self.act(self.norm(self.conv(x)))

    def convert_to_deploy(self) -> None:
        """Fuse conv and batchnorm for deployment."""
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self._ch_in,
                self._ch_out,
                self._kernel_size,
                self._stride,
                groups=self._groups,
                padding=self._padding,
                bias=True,
            )
        kernel, bias = self._get_fused_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        delattr(self, "conv")
        delattr(self, "norm")

    def _get_fused_kernel_bias(self) -> tuple[Tensor, Tensor]:
        """Get fused kernel and bias from conv and batchnorm."""
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class VGGBlock(nn.Module):
    """VGG-style block with parallel 3x3 and 1x1 convolutions.

    Can be converted to a single 3x3 conv for deployment.

    Args:
        ch_in: Input channels.
        ch_out: Output channels.
        act: Activation function class.
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        act: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if hasattr(self, "conv"):
            return self.act(self.conv(x))
        return self.act(self.conv1(x) + self.conv2(x))

    def convert_to_deploy(self) -> None:
        """Fuse parallel branches into single conv for deployment."""
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        kernel, bias = self._get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        delattr(self, "conv1")
        delattr(self, "conv2")

    def _get_equivalent_kernel_bias(self) -> tuple[Tensor, Tensor]:
        """Get equivalent 3x3 kernel and bias."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + f.pad(kernel1x1, [1, 1, 1, 1]), bias3x3 + bias1x1

    def _fuse_bn_tensor(self, branch: ConvNormLayer) -> tuple[Tensor, Tensor]:
        """Fuse batchnorm into conv weights."""
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


# =============================================================================
# CSP Layers
# =============================================================================


class CSPLayerV2(nn.Module):
    """Cross Stage Partial Layer V2.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_blocks: Number of bottleneck blocks.
        expansion: Channel expansion ratio.
        bias: Whether to use bias.
        act: Activation function class.
        bottletype: Bottleneck block type.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        bias: bool = False,
        act: Callable[..., nn.Module] = nn.SiLU,
        bottletype: type[nn.Module] = VGGBlock,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayerFusable(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayerFusable(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)]
        )
        self.conv3: nn.Module = (
            ConvNormLayerFusable(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
            if hidden_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x_1 = self.bottlenecks(self.conv1(x))
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class CSPLayer2(nn.Module):
    """Cross Stage Partial Layer with chunk-based split.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_blocks: Number of bottleneck blocks.
        expansion: Channel expansion ratio.
        bias: Whether to use bias.
        act: Activation function class.
        bottletype: Bottleneck block type.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        bias: bool = False,
        act: Callable[..., nn.Module] = nn.SiLU,
        bottletype: type[nn.Module] = VGGBlock,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayerFusable(in_channels, hidden_channels * 2, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)]
        )
        self.conv3: nn.Module = (
            ConvNormLayerFusable(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
            if hidden_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        y = list(self.conv1(x).chunk(2, 1))
        return self.conv3(y[0] + self.bottlenecks(y[1]))


# =============================================================================
# Downsampling Modules
# =============================================================================


class SCDown(nn.Module):
    """Spatial-Channel Downsampling module.

    Args:
        c1: Input channels.
        c2: Output channels.
        k: Kernel size.
        s: Stride.
        normalization: Normalization layer builder.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int,
        s: int,
        normalization: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.cv1 = Conv2dModule(
            c1,
            c2,
            1,
            1,
            normalization=build_norm_layer(normalization, num_features=c2),
            activation=None,
        )
        self.cv2 = Conv2dModule(
            c2,
            c2,
            k,
            s,
            padding=auto_pad(kernel_size=k),
            groups=c2,
            normalization=build_norm_layer(normalization, num_features=c2),
            activation=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.cv2(self.cv1(x))


class SCDownFusable(nn.Module):
    """Fusable Spatial-Channel Downsampling module.

    Args:
        c1: Input channels.
        c2: Output channels.
        k: Kernel size.
        s: Stride.
    """

    def __init__(self, c1: int, c2: int, k: int, s: int) -> None:
        super().__init__()
        self.cv1 = ConvNormLayerFusable(c1, c2, 1, 1)
        self.cv2 = ConvNormLayerFusable(c2, c2, k, s, groups=c2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.cv2(self.cv1(x))


# =============================================================================
# RepNCSPELAN Blocks
# =============================================================================


class RepNCSPELAN4(nn.Module):
    """RepNCSPELAN4 block (GELAN-style).

    Args:
        c1: Input channels.
        c2: Output channels.
        c3: Internal channels.
        c4: Bottleneck channels.
        num_blocks: Number of blocks.
        bias: Whether to use bias.
        activation: Activation layer builder.
        normalization: Normalization layer builder.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        num_blocks: int = 3,
        bias: bool = False,
        activation: Callable[..., nn.Module] | None = None,
        normalization: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.c = c3 // 2

        self.cv1 = Conv2dModule(
            c1,
            c3,
            1,
            1,
            bias=bias,
            activation=build_activation_layer(activation),
            normalization=build_norm_layer(normalization, num_features=c3),
        )
        self.cv2 = nn.Sequential(
            CSPRepLayer(c3 // 2, c4, num_blocks, 1, bias=bias, activation=activation, normalization=normalization),
            Conv2dModule(
                c4,
                c4,
                3,
                1,
                padding=auto_pad(kernel_size=3),
                bias=bias,
                activation=build_activation_layer(activation),
                normalization=build_norm_layer(normalization, num_features=c4),
            ),
        )
        self.cv3 = nn.Sequential(
            CSPRepLayer(c4, c4, num_blocks, 1, bias=bias, activation=activation, normalization=normalization),
            Conv2dModule(
                c4,
                c4,
                3,
                1,
                padding=auto_pad(kernel_size=3),
                bias=bias,
                activation=build_activation_layer(activation),
                normalization=build_norm_layer(normalization, num_features=c4),
            ),
        )
        self.cv4 = Conv2dModule(
            c3 + (2 * c4),
            c2,
            1,
            1,
            bias=bias,
            activation=build_activation_layer(activation),
            normalization=build_norm_layer(normalization, num_features=c2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class RepNCSPELAN5(nn.Module):
    """RepNCSPELAN5 block (DEIM-style, fusable implementation).

    Args:
        c1: Input channels.
        c2: Output channels.
        c3: Internal channels.
        c4: Bottleneck channels.
        num_blocks: Number of blocks.
        bias: Whether to use bias.
        act: Activation function class.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        num_blocks: int = 3,
        bias: bool = False,
        act: Callable[..., nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayerFusable(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer2(c3 // 2, c4, num_blocks, 1, bias=bias, act=act, bottletype=VGGBlock))
        self.cv3 = nn.Sequential(CSPLayer2(c4, c4, num_blocks, 1, bias=bias, act=act, bottletype=VGGBlock))
        self.cv4 = ConvNormLayerFusable(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# =============================================================================
# Main Hybrid Encoder Module
# =============================================================================


class HybridEncoderModule(nn.Module):
    """Unified Hybrid Encoder for D-FINE and DEIM detection models.

    Combines transformer encoder with FPN and PAN for multi-scale feature fusion.

    Args:
        in_channels: List of input channel sizes for each feature level.
        feat_strides: List of stride values for each feature level.
        hidden_dim: Hidden dimension for the encoder.
        nhead: Number of attention heads.
        dim_feedforward: Feedforward dimension in transformer.
        dropout: Dropout rate.
        enc_activation: Activation for transformer encoder.
        use_encoder_idx: Indices of feature levels to apply transformer encoder.
        num_encoder_layers: Number of transformer encoder layers.
        pe_temperature: Temperature for positional encoding.
        expansion: Channel expansion factor.
        depth_mult: Depth multiplier for CSP blocks.
        activation: Activation function class for FPN/PAN blocks.
        normalization: Normalization layer builder.
        eval_spatial_size: Spatial size for evaluation (caches positional embeddings).
        fuse_op: Feature fusion operation ('cat' or 'sum').
        use_fusable_layers: Whether to use fusable layers (for DEIM models).
    """

    def __init__(
        self,
        in_channels: list[int] | None = None,
        feat_strides: list[int] | None = None,
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        enc_activation: Callable[..., nn.Module] = nn.GELU,
        use_encoder_idx: list[int] | None = None,
        num_encoder_layers: int = 1,
        pe_temperature: float = 10000.0,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        activation: Callable[..., nn.Module] = nn.SiLU,
        normalization: Callable[..., nn.Module] | None = None,
        eval_spatial_size: tuple[int, int] | None = None,
        fuse_op: Literal["cat", "sum"] = "cat",
        use_fusable_layers: bool = False,
    ) -> None:
        super().__init__()

        # Set defaults
        if in_channels is None:
            in_channels = [512, 1024, 2048]
        if feat_strides is None:
            feat_strides = [8, 16, 32]
        if use_encoder_idx is None:
            use_encoder_idx = [2]
        if normalization is None:
            normalization = partial(build_norm_layer, nn.BatchNorm2d, layer_name="norm")

        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.fuse_op = fuse_op
        self.out_channels = [hidden_dim] * len(in_channels)
        self.out_strides = feat_strides

        # Build input projection
        self.input_proj = self._build_input_proj(in_channels, hidden_dim, use_fusable_layers)

        # Build transformer encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_activation,
        )
        self.encoder = nn.ModuleList(
            [TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))]
        )

        # Build FPN and PAN
        self._build_fpn_pan(
            in_channels, hidden_dim, expansion, depth_mult, activation, normalization, fuse_op, use_fusable_layers
        )

        self._reset_parameters()

    def _build_input_proj(
        self,
        in_channels: list[int],
        hidden_dim: int,
        use_fusable_layers: bool,
    ) -> nn.ModuleList:
        """Build input projection layers."""
        input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if use_fusable_layers and in_channel == hidden_dim:
                input_proj.append(nn.Identity())
            else:
                input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                                ("norm", nn.BatchNorm2d(hidden_dim)),
                            ]
                        )
                    )
                )
        return input_proj

    def _build_fpn_pan(
        self,
        in_channels: list[int],
        hidden_dim: int,
        expansion: float,
        depth_mult: float,
        activation: Callable[..., nn.Module],
        normalization: Callable[..., nn.Module],
        fuse_op: str,
        use_fusable_layers: bool,
    ) -> None:
        """Build FPN and PAN layers."""
        num_levels = len(in_channels)
        input_dim = hidden_dim if fuse_op == "sum" else hidden_dim * 2
        num_blocks = round(3 * depth_mult)
        c4 = round(expansion * hidden_dim // 2)

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()

        for _ in range(num_levels - 1):
            if use_fusable_layers:
                # DEIM-style fusable layers
                self.lateral_convs.append(ConvNormLayerFusable(hidden_dim, hidden_dim, 1, 1))
                self.fpn_blocks.append(
                    RepNCSPELAN5(input_dim, hidden_dim, hidden_dim * 2, c4, num_blocks, act=activation)
                )
                self.downsample_convs.append(nn.Sequential(SCDownFusable(hidden_dim, hidden_dim, 3, 2)))
                self.pan_blocks.append(
                    RepNCSPELAN5(input_dim, hidden_dim, hidden_dim * 2, c4, num_blocks, act=activation)
                )
            else:
                # D-FINE style with getitune layers
                self.lateral_convs.append(
                    Conv2dModule(
                        hidden_dim,
                        hidden_dim,
                        1,
                        1,
                        normalization=build_norm_layer(normalization, num_features=hidden_dim),
                        activation=None,
                    )
                )
                self.fpn_blocks.append(
                    RepNCSPELAN4(
                        hidden_dim * 2,
                        hidden_dim,
                        hidden_dim * 2,
                        c4,
                        num_blocks,
                        activation=activation,
                        normalization=normalization,
                    )
                )
                self.downsample_convs.append(
                    nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, normalization=normalization))
                )
                self.pan_blocks.append(
                    RepNCSPELAN4(
                        hidden_dim * 2,
                        hidden_dim,
                        hidden_dim * 2,
                        c4,
                        num_blocks,
                        activation=activation,
                        normalization=normalization,
                    )
                )

    def _reset_parameters(self) -> None:
        """Initialize cached positional embeddings for evaluation."""
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
    ) -> Tensor:
        """Build 2D sinusoidal-cosine position embedding.

        Args:
            w: Width of the feature map.
            h: Height of the feature map.
            embed_dim: Embedding dimension (must be divisible by 4).
            temperature: Temperature for positional encoding.

        Returns:
            Position embedding tensor of shape (1, h*w, embed_dim).
        """
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

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        """Forward pass.

        Args:
            feats: List of feature tensors from backbone.

        Returns:
            List of fused multi-scale feature tensors.
        """
        if len(feats) != len(self.in_channels):
            msg = f"Input feature size {len(feats)} does not match expected {len(self.in_channels)}"
            raise ValueError(msg)

        # Project input features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Apply transformer encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)

                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(
                        src_flatten.device
                    )
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}").to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # Top-down FPN
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]

            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high

            upsample_feat = f.interpolate(feat_high, scale_factor=2.0, mode="nearest")

            if self.fuse_op == "sum":
                fused_feat = upsample_feat + feat_low
            else:
                fused_feat = torch.concat([upsample_feat, feat_low], dim=1)

            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](fused_feat)
            inner_outs.insert(0, inner_out)

        # Bottom-up PAN
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]

            downsample_feat = self.downsample_convs[idx](feat_low)

            if self.fuse_op == "sum":
                fused_feat = downsample_feat + feat_high
            else:
                fused_feat = torch.concat([downsample_feat, feat_high], dim=1)

            out = self.pan_blocks[idx](fused_feat)
            outs.append(out)

        return outs


# =============================================================================
# Factory Class
# =============================================================================


class HybridEncoder:
    """Factory class for creating HybridEncoder instances.

    Supports D-FINE (dfine_*), DEIM-DFINE (deim_dfine_*), and DEIMv2 (deimv2_*) models.
    """

    encoder_cfg: ClassVar[dict[str, dict[str, Any]]] = {
        # D-FINE models (use concat fusion, getitune layers)
        "dfine_hgnetv2_n": {
            "in_channels": [512, 1024],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "use_encoder_idx": [1],
            "dim_feedforward": 512,
            "expansion": 0.34,
            "depth_mult": 0.5,
            "eval_spatial_size": [640, 640],
        },
        "dfine_hgnetv2_s": {
            "in_channels": [256, 512, 1024],
            "hidden_dim": 256,
            "expansion": 0.5,
            "depth_mult": 0.34,
            "eval_spatial_size": [640, 640],
        },
        "dfine_hgnetv2_m": {
            "in_channels": [384, 768, 1536],
            "hidden_dim": 256,
            "depth_mult": 0.67,
            "eval_spatial_size": [640, 640],
        },
        "dfine_hgnetv2_l": {},
        "dfine_hgnetv2_x": {
            "hidden_dim": 384,
            "dim_feedforward": 2048,
        },
        # DEIM-DFINE models (use concat fusion, getitune layers)
        "deim_dfine_hgnetv2_n": {
            "in_channels": [512, 1024],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "use_encoder_idx": [1],
            "dim_feedforward": 512,
            "expansion": 0.34,
            "depth_mult": 0.5,
            "eval_spatial_size": [640, 640],
        },
        "deim_dfine_hgnetv2_s": {
            "in_channels": [256, 512, 1024],
            "hidden_dim": 256,
            "expansion": 0.5,
            "depth_mult": 0.34,
            "eval_spatial_size": [640, 640],
        },
        "deim_dfine_hgnetv2_m": {
            "in_channels": [384, 768, 1536],
            "hidden_dim": 256,
            "depth_mult": 0.67,
            "eval_spatial_size": [640, 640],
        },
        "deim_dfine_hgnetv2_l": {},
        "deim_dfine_hgnetv2_x": {
            "hidden_dim": 384,
            "dim_feedforward": 2048,
        },
        # DEIMv2 models (use sum fusion, fusable layers)
        "deimv2_x": {
            "in_channels": [256, 256, 256],
            "hidden_dim": 256,
            "dim_feedforward": 1024,
            "expansion": 1.25,
            "depth_mult": 1.37,
            "fuse_op": "sum",
            "use_fusable_layers": True,
        },
        "deimv2_l": {
            "in_channels": [224, 224, 224],
            "hidden_dim": 224,
            "dim_feedforward": 896,
            "fuse_op": "sum",
            "use_fusable_layers": True,
        },
        "deimv2_m": {
            "in_channels": [256, 256, 256],
            "depth_mult": 1.0,
            "expansion": 0.67,
            "hidden_dim": 256,
            "dim_feedforward": 512,
            "fuse_op": "sum",
            "use_fusable_layers": True,
        },
        "deimv2_s": {
            "in_channels": [192, 192, 192],
            "depth_mult": 0.67,
            "expansion": 0.34,
            "hidden_dim": 192,
            "dim_feedforward": 512,
            "fuse_op": "sum",
            "use_fusable_layers": True,
        },
        # EdgeCrafter models (use sum fusion, fusable layers)
        "edgecrafter_s": {
            "in_channels": [192, 192, 192],
            "hidden_dim": 192,
            "dim_feedforward": 512,
            "depth_mult": 0.67,
            "expansion": 0.34,
            "fuse_op": "sum",
            "use_fusable_layers": True,
        },
        "edgecrafter_m": {
            "in_channels": [256, 256, 256],
            "hidden_dim": 256,
            "dim_feedforward": 512,
            "depth_mult": 0.67,
            "expansion": 0.75,
            "fuse_op": "sum",
            "use_fusable_layers": True,
        },
        "edgecrafter_l": {
            "in_channels": [256, 256, 256],
            "hidden_dim": 256,
            "dim_feedforward": 1024,
            "depth_mult": 1.0,
            "expansion": 0.75,
            "fuse_op": "sum",
            "use_fusable_layers": True,
        },
        "edgecrafter_x": {
            "in_channels": [256, 256, 256],
            "hidden_dim": 256,
            "dim_feedforward": 2048,
            "depth_mult": 1.0,
            "expansion": 1.5,
            "fuse_op": "sum",
            "use_fusable_layers": True,
        },
    }

    def __new__(cls, model_name: str) -> HybridEncoderModule:
        """Create a HybridEncoder instance.

        Args:
            model_name: Model configuration name.

        Returns:
            Configured HybridEncoderModule instance.

        Raises:
            KeyError: If model_name is not in encoder_cfg.
        """
        if model_name not in cls.encoder_cfg:
            msg = f"Model type '{model_name}' is not supported. Available: {list(cls.encoder_cfg.keys())}"
            raise KeyError(msg)
        return HybridEncoderModule(**cls.encoder_cfg[model_name])
