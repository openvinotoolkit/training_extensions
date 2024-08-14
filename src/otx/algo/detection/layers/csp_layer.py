# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation of CSPLayer copied from mmdet.models.layers.csp_layer.py."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import Tensor, nn

from otx.algo.detection.layers import ChannelAttention
from otx.algo.modules.activation import Swish, build_activation_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.conv_module import Conv2dModule, DepthwiseSeparableConvModule
from otx.algo.modules.norm import build_norm_layer


class DarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): The kernel size of the convolution.
            Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out.
            Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(nn.BatchNorm2d, momentum=0.03, eps=0.001)``.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``Swish``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        normalization: Callable[..., nn.Module] = partial(nn.BatchNorm2d, momentum=0.03, eps=0.001),
        activation: Callable[..., nn.Module] = Swish,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else Conv2dModule
        self.conv1 = Conv2dModule(
            in_channels,
            hidden_channels,
            1,
            normalization=build_norm_layer(normalization, num_features=hidden_channels),
            activation=build_activation_layer(activation),
        )
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            normalization=build_norm_layer(normalization, num_features=out_channels),
            activation=build_activation_layer(activation),
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        return out


class CSPNeXtBlock(BaseModule):
    """The basic bottleneck block used in CSPNeXt.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): Expand ratio of the hidden channel. Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out. Only works
            when in_channels == out_channels. Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        kernel_size (int): The kernel size of the second convolution layer.
            Defaults to 5.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to ``partial(nn.BatchNorm2d, momentum=0.03, eps=0.001)``.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.SiLU``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        kernel_size: int = 5,
        normalization: Callable[..., nn.Module] = partial(nn.BatchNorm2d, momentum=0.03, eps=0.001),
        activation: Callable[..., nn.Module] = nn.SiLU,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else Conv2dModule
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            normalization=build_norm_layer(normalization, num_features=hidden_channels),
            activation=build_activation_layer(activation),
        )
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            normalization=build_norm_layer(normalization, num_features=out_channels),
            activation=build_activation_layer(activation),
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        return out


class RepVggBlock(nn.Module):
    """RepVggBlock.

    Args:
        ch_in (int): The input channels of this Module.
        ch_out (int): The output channels of this Module.
        activation (Callable[..., nn.Module] | None): Activation layer module.
            Defaults to None.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to None.
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        activation: Callable[..., nn.Module] | None = None,
        normalization: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Initialize RepVggBlock."""
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = Conv2dModule(
            ch_in,
            ch_out,
            3,
            1,
            padding=1,
            normalization=build_norm_layer(normalization, num_features=ch_out),
            activation=None,
        )
        self.conv2 = Conv2dModule(
            ch_in,
            ch_out,
            1,
            1,
            normalization=build_norm_layer(normalization, num_features=ch_out),
            activation=None,
        )
        self.act = activation() if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        y = self.conv(x) if hasattr(self, "conv") else self.conv1(x) + self.conv2(x)
        return self.act(y)

    def get_equivalent_kernel_bias(self) -> tuple[Tensor, Tensor]:
        """Get the equivalent kernel and bias of the block."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1: Tensor | None) -> Tensor:
        """Pad the 1x1 kernel to 3x3 kernel."""
        if kernel1x1 is None:
            return 0
        return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: Conv2dModule) -> tuple[float, float]:
        """Fuse the BN layer to the convolution layer."""
        if branch is None or branch.norm_layer is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm_layer.running_mean
        running_var = branch.norm_layer.running_var
        gamma = branch.norm_layer.weight
        beta = branch.norm_layer.bias
        eps = branch.norm_layer.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPLayer(BaseModule):
    """Cross Stage Partial Layer.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1.
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
        use_cspnext_block (bool): Whether to use CSPNeXt block.
            Defaults to False.
        use_depthwise (bool): Whether to use depthwise separable convolution in
            blocks. Defaults to False.
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(nn.BatchNorm2d, momentum=0.03, eps=0.001)``.
        activation (Callable[..., nn.Module] | None): Activation layer module.
            Defaults to ``Swish``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        use_cspnext_block: bool = False,
        channel_attention: bool = False,
        normalization: Callable[..., nn.Module] = partial(nn.BatchNorm2d, momentum=0.03, eps=0.001),
        activation: Callable[..., nn.Module] | None = Swish,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = Conv2dModule(
            in_channels,
            mid_channels,
            1,
            normalization=build_norm_layer(normalization, num_features=mid_channels),
            activation=build_activation_layer(activation),
        )
        self.short_conv = Conv2dModule(
            in_channels,
            mid_channels,
            1,
            normalization=build_norm_layer(normalization, num_features=mid_channels),
            activation=build_activation_layer(activation),
        )
        self.final_conv = Conv2dModule(
            2 * mid_channels,
            out_channels,
            1,
            normalization=build_norm_layer(normalization, num_features=out_channels),
            activation=build_activation_layer(activation),
        )

        self.blocks = nn.Sequential(
            *[
                block(
                    mid_channels,
                    mid_channels,
                    1.0,
                    add_identity,
                    use_depthwise,
                    normalization=normalization,
                    activation=activation,
                )
                for _ in range(num_blocks)
            ],
        )
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class CSPRepLayer(nn.Module):
    """Cross Stage Partial Layer with RepVGGBlock.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        num_blocks (int): Number of blocks. Defaults to 3.
        expansion (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 1.0.
        bias (bool): Whether to use bias in the convolution layer.
            Defaults to False.
        activation (Callable[..., nn.Module] | None): Activation layer module.
            Defaults to None.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        bias: bool = False,
        activation: Callable[..., nn.Module] | None = None,
        normalization: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Initialize CSPRepLayer."""
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv2dModule(
            in_channels,
            hidden_channels,
            1,
            1,
            bias=bias,
            normalization=build_norm_layer(normalization, num_features=hidden_channels),
            activation=build_activation_layer(activation),
        )
        self.conv2 = Conv2dModule(
            in_channels,
            hidden_channels,
            1,
            1,
            bias=bias,
            normalization=build_norm_layer(normalization, num_features=hidden_channels),
            activation=build_activation_layer(activation),
        )
        self.bottlenecks = nn.Sequential(
            *[
                RepVggBlock(
                    hidden_channels,
                    hidden_channels,
                    activation=activation,
                    normalization=normalization,
                )
                for _ in range(num_blocks)
            ],
        )
        if hidden_channels != out_channels:
            self.conv3 = Conv2dModule(
                hidden_channels,
                out_channels,
                1,
                1,
                bias=bias,
                normalization=build_norm_layer(normalization, num_features=out_channels),
                activation=build_activation_layer(activation),
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)
