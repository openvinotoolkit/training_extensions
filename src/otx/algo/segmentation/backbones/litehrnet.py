# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""HRNet network modules for base backbone.

Modified from:
- https://github.com/HRNet/Lite-HRNet
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.utils.checkpoint as cp
from torch import nn
from torch.nn import functional

from otx.algo.modules import Conv2dModule, build_norm_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.segmentation.modules import (
    AsymmetricPositionAttentionModule,
    IterativeAggregator,
    LocalAttentionModule,
    channel_shuffle,
)
from otx.algo.utils.mmengine_utils import load_checkpoint_to_model, load_from_http


class NeighbourSupport(nn.Module):
    """Neighbour support module.

    Args:
        channels (int): Number of input channels.
        kernel_size (int): Kernel size for convolutional layers. Default is 3.
        key_ratio (int): Ratio of input channels to key channels. Default is 8.
        value_ratio (int): Ratio of input channels to value channels. Default is 8.
        normalization_callable (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to None.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        key_ratio: int = 8,
        value_ratio: int = 8,
        normalization_callable: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        self.in_channels = channels
        self.key_channels = int(channels / key_ratio)
        self.value_channels = int(channels / value_ratio)
        self.kernel_size = kernel_size

        self.key = nn.Sequential(
            Conv2dModule(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                normalization=build_norm_layer(normalization_callable, num_features=self.key_channels),
                activation_callable=nn.ReLU,
            ),
            Conv2dModule(
                self.key_channels,
                self.key_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=(self.kernel_size - 1) // 2,
                groups=self.key_channels,
                normalization=build_norm_layer(normalization_callable, num_features=self.key_channels),
                activation_callable=None,
            ),
            Conv2dModule(
                in_channels=self.key_channels,
                out_channels=self.kernel_size * self.kernel_size,
                kernel_size=1,
                stride=1,
                normalization=build_norm_layer(
                    normalization_callable,
                    num_features=self.kernel_size * self.kernel_size,
                ),
                activation_callable=None,
            ),
        )
        self.value = nn.Sequential(
            Conv2dModule(
                in_channels=self.in_channels,
                out_channels=self.value_channels,
                kernel_size=1,
                stride=1,
                normalization=build_norm_layer(normalization_callable, num_features=self.value_channels),
                activation_callable=None,
            ),
            nn.Unfold(kernel_size=self.kernel_size, stride=1, padding=1),
        )
        self.out_conv = Conv2dModule(
            in_channels=self.value_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            normalization=build_norm_layer(normalization_callable, num_features=self.in_channels),
            activation_callable=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        h, w = (int(_) for _ in x.size()[-2:])

        key = self.key(x).view(-1, 1, self.kernel_size**2, h, w)
        weights = torch.softmax(key, dim=2)

        value = self.value(x).view(-1, self.value_channels, self.kernel_size**2, h, w)
        y = torch.sum(weights * value, dim=2)
        y = self.out_conv(y)

        return x + y


class CrossResolutionWeighting(nn.Module):
    """Cross resolution weighting.

    Args:
        channels (list[int]): Number of channels for each stage.
        ratio (int): Reduction ratio of the bottleneck block.
        normalization_callable (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to None.
        activation_callable (Callable[..., nn.Module] | tuple[Callable[..., nn.Module], Callable[..., nn.Module]]): \
            Activation layer module or a tuple of activation layer modules.
            Defaults to ``(nn.ReLU, nn.Sigmoid)``.
    """

    def __init__(
        self,
        channels: list[int],
        ratio: int = 16,
        normalization_callable: Callable[..., nn.Module] | None = None,
        activation_callable: Callable[..., nn.Module] | tuple[Callable[..., nn.Module], Callable[..., nn.Module]] = (
            nn.ReLU,
            nn.Sigmoid,
        ),
    ) -> None:
        super().__init__()

        if callable(activation_callable):
            activation_callable = (activation_callable, activation_callable)

        if len(activation_callable) != 2:
            msg = "activation_callable must be a callable or a tuple of callables of length 2."
            raise ValueError(msg)

        self.channels = channels
        total_channel = sum(channels)

        self.conv1 = Conv2dModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            normalization=build_norm_layer(normalization_callable, num_features=int(total_channel / ratio)),
            activation_callable=activation_callable[0],
        )
        self.conv2 = Conv2dModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            normalization=build_norm_layer(normalization_callable, num_features=total_channel),
            activation_callable=activation_callable[1],
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward."""
        min_size = [int(_) for _ in x[-1].size()[-2:]]

        out = [functional.adaptive_avg_pool2d(s, min_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)

        return [s * functional.interpolate(a, size=s.size()[-2:], mode="nearest") for s, a in zip(x, out)]


class SpatialWeighting(nn.Module):
    """Spatial weighting.

    Args:
        channels (int): Number of input channels.
        ratio (int): Reduction ratio for the bottleneck block. Default: 16.
        activation_callable (Callable[..., nn.Module] | tuple[Callable[..., nn.Module], Callable[..., nn.Module]]): \
            Activation layer module or a tuple of activation layer modules.
            If a single module is provided, it will be used for both activation layers.
            Defaults to ``(nn.ReLU, nn.Sigmoid)``.

    Raises:
        ValueError: activation_callable must be a callable or a tuple of callables of length 2.
        TypeError: If activation_callable is not a callable or a tuple of callables.
    """

    def __init__(
        self,
        channels: int,
        ratio: int = 16,
        activation_callable: Callable[..., nn.Module] | tuple[Callable[..., nn.Module], Callable[..., nn.Module]] = (
            nn.ReLU,
            nn.Sigmoid,
        ),
        **kwargs,
    ) -> None:
        super().__init__()

        if callable(activation_callable):
            activation_callable = (activation_callable, activation_callable)

        if len(activation_callable) != 2:
            msg = "activation_callable must be a callable or a tuple of callables of length 2."
            raise ValueError(msg)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = Conv2dModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            activation_callable=activation_callable[0],
        )
        self.conv2 = Conv2dModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            activation_callable=activation_callable[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)

        return x * out


class SpatialWeightingV2(nn.Module):
    """SpatialWeightingV2.

    The original repo: https://github.com/DeLightCMU/PSA.

    Args:
        channels (int): Number of input channels.
        ratio (int): Reduction ratio of internal channels.
        normalization_callable (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to None.
        enable_norm (bool): Whether to enable normalization layers.
    """

    def __init__(
        self,
        channels: int,
        ratio: int = 16,
        normalization_callable: Callable[..., nn.Module] | None = None,
        enable_norm: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = channels
        self.internal_channels = int(channels / ratio)

        # channel-only branch
        self.v_channel = Conv2dModule(
            in_channels=self.in_channels,
            out_channels=self.internal_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            normalization=build_norm_layer(normalization_callable, num_features=self.internal_channels)
            if enable_norm
            else None,
            activation_callable=None,
        )
        self.q_channel = Conv2dModule(
            in_channels=self.in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False,
            normalization=build_norm_layer(normalization_callable, num_features=1) if enable_norm else None,
            activation_callable=None,
        )
        self.out_channel = Conv2dModule(
            in_channels=self.internal_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            normalization=build_norm_layer(normalization_callable, num_features=self.in_channels),
            activation_callable=nn.Sigmoid,
        )

        # spatial-only branch
        self.v_spatial = Conv2dModule(
            in_channels=self.in_channels,
            out_channels=self.internal_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            normalization=build_norm_layer(normalization_callable, num_features=self.internal_channels)
            if enable_norm
            else None,
            activation_callable=None,
        )
        self.q_spatial = Conv2dModule(
            in_channels=self.in_channels,
            out_channels=self.internal_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            normalization=build_norm_layer(normalization_callable, num_features=self.internal_channels)
            if enable_norm
            else None,
            activation_callable=None,
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

    def _channel_weighting(self, x: torch.Tensor) -> torch.Tensor:
        """_channel_weighting.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        h, w = (int(_) for _ in x.size()[-2:])

        v = self.v_channel(x).view(-1, self.internal_channels, h * w)

        q = self.q_channel(x).view(-1, h * w, 1)
        q = torch.softmax(q, dim=1)

        y = torch.matmul(v, q)
        y = y.view(-1, self.internal_channels, 1, 1)
        y = self.out_channel(y)

        return x * y

    def _spatial_weighting(self, x: torch.Tensor) -> torch.Tensor:
        """_spatial_weighting.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        h, w = (int(_) for _ in x.size()[-2:])

        v = self.v_spatial(x)
        v = v.view(-1, self.internal_channels, h * w)

        q = self.q_spatial(x)
        q = self.global_avgpool(q)
        q = torch.softmax(q, dim=1)
        q = q.view(-1, 1, self.internal_channels)

        y = torch.matmul(q, v)
        y = y.view(-1, 1, h, w)
        y = torch.sigmoid(y)

        return x * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        y_channel = self._channel_weighting(x)
        y_spatial = self._spatial_weighting(x)

        return y_channel + y_spatial


class ConditionalChannelWeighting(nn.Module):
    """Conditional channel weighting module.

    Args:
        in_channels (list[int]): Number of input channels for each input feature map.
        stride (int): Stride used in the first convolutional layer.
        reduce_ratio (int): Reduction ratio used in the cross-resolution weighting module.
        normalization_callable (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``nn.BatchNorm2d``.
        with_cp (bool): Whether to use checkpointing to save memory.
        dropout (float | None): Dropout probability used in the depthwise convolutional layers.
        weighting_module_version (str): Version of the spatial weighting module to use.
        neighbour_weighting (bool): Whether to use the neighbour support module.
        dw_ksize (int): Kernel size used in the depthwise convolutional layers.

    Raises:
        ValueError: If stride is not 1 or 2.
    """

    def __init__(
        self,
        in_channels: list[int],
        stride: int,
        reduce_ratio: int,
        normalization_callable: Callable[..., nn.Module] = nn.BatchNorm2d,
        with_cp: bool = False,
        dropout: float | None = None,
        weighting_module_version: str = "v1",
        neighbour_weighting: bool = False,
        dw_ksize: int = 3,
    ) -> None:
        super().__init__()

        self.with_cp = with_cp
        self.stride = stride
        if stride not in [1, 2]:
            msg = "stride must be 1 or 2."
            raise ValueError(msg)

        spatial_weighting_module = SpatialWeighting if weighting_module_version == "v1" else SpatialWeightingV2
        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio,
            normalization_callable=normalization_callable,
        )
        self.depthwise_convs = nn.ModuleList(
            [
                Conv2dModule(
                    channel,
                    channel,
                    kernel_size=dw_ksize,
                    stride=self.stride,
                    padding=dw_ksize // 2,
                    groups=channel,
                    normalization=build_norm_layer(normalization_callable, num_features=channel),
                    activation_callable=None,
                )
                for channel in branch_channels
            ],
        )
        self.spatial_weighting = nn.ModuleList(
            [
                spatial_weighting_module(  # type: ignore[call-arg]
                    channels=channel,
                    ratio=4,
                    normalization_callable=normalization_callable,
                    enable_norm=True,
                )
                for channel in branch_channels
            ],
        )

        self.neighbour_weighting = None
        if neighbour_weighting:
            self.neighbour_weighting = nn.ModuleList(
                [
                    NeighbourSupport(
                        channel,
                        kernel_size=3,
                        key_ratio=8,
                        value_ratio=4,
                        normalization_callable=normalization_callable,
                    )
                    for channel in branch_channels
                ],
            )

        self.dropout = None
        if dropout is not None and dropout > 0.0:
            self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in branch_channels])

    def _inner_forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """_inner_forward.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            list[torch.Tensor]: Output tensor.
        """
        x = [s.chunk(2, dim=1) for s in x]
        x1 = [s[0] for s in x]
        x2 = [s[1] for s in x]

        x2 = self.cross_resolution_weighting(x2)
        x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]

        if self.neighbour_weighting is not None:
            x2 = [nw(s) for s, nw in zip(x2, self.neighbour_weighting)]

        x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

        if self.dropout is not None:
            x2 = [dropout(s) for s, dropout in zip(x2, self.dropout)]

        out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]

        return [channel_shuffle(s, 2) for s in out]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward."""
        return cp.checkpoint(self._inner_forward, x) if self.with_cp and x.requires_grad else self._inner_forward(x)


class Stem(nn.Module):
    """Stem.

    Args:
        in_channels (int): Number of input image channels. Typically 3.
        stem_channels (int): Number of output channels of the stem layer.
        out_channels (int): Number of output channels of the backbone network.
        expand_ratio (int): Expansion ratio of the internal channels.
        normalization_callable (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to ``nn.BatchNorm2d``.
        with_cp (bool): Use checkpointing to save memory during forward pass.
        num_stages (int): Number of stages in the backbone network.
        strides (tuple[int, int]): Strides of the first and subsequent stages.
        extra_stride (bool): Use an extra stride in the second stage.
        input_norm (bool): Use instance normalization on the input image.

    Raises:
        TypeError: If strides is not a tuple or list.
        ValueError: If len(strides) is not equal to num_stages + 1.
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        out_channels: int,
        expand_ratio: int,
        normalization_callable: Callable[..., nn.Module] = nn.BatchNorm2d,
        with_cp: bool = False,
        strides: tuple[int, int] = (2, 2),
        extra_stride: bool = False,
        input_norm: bool = False,
    ) -> None:
        """Stem initialization."""
        super().__init__()

        if not isinstance(strides, (tuple, list)):
            msg = "strides must be tuple or list."
            raise TypeError(msg)
        if len(strides) != 2:
            msg = "len(strides) must equal to 2."
            raise ValueError(msg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization_callable = normalization_callable
        self.with_cp = with_cp

        self.input_norm = None
        if input_norm:
            self.input_norm = nn.InstanceNorm2d(in_channels)

        self.conv1 = Conv2dModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=strides[0],
            padding=1,
            normalization=build_norm_layer(self.normalization_callable, num_features=stem_channels),
            activation_callable=nn.ReLU,
        )

        self.conv2 = None
        if extra_stride:
            self.conv2 = Conv2dModule(
                in_channels=stem_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                normalization=build_norm_layer(self.normalization_callable, num_features=stem_channels),
                activation_callable=nn.ReLU,
            )

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            Conv2dModule(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=strides[1],
                padding=1,
                groups=branch_channels,
                normalization=build_norm_layer(normalization_callable, num_features=branch_channels),
                activation_callable=None,
            ),
            Conv2dModule(
                branch_channels,
                inc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                normalization=build_norm_layer(normalization_callable, num_features=inc_channels),
                activation_callable=nn.ReLU,
            ),
        )

        self.expand_conv = Conv2dModule(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            normalization=build_norm_layer(normalization_callable, num_features=mid_channels),
            activation_callable=nn.ReLU,
        )
        self.depthwise_conv = Conv2dModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=strides[1],
            padding=1,
            groups=mid_channels,
            normalization=build_norm_layer(normalization_callable, num_features=mid_channels),
            activation_callable=None,
        )
        self.linear_conv = Conv2dModule(
            mid_channels,
            branch_channels if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            normalization=build_norm_layer(
                normalization_callable,
                num_features=branch_channels if stem_channels == self.out_channels else stem_channels,
            ),
            activation_callable=nn.ReLU,
        )

    def _inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """_inner_forward.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.input_norm is not None:
            x = self.input_norm(x)

        x = self.conv1(x)
        if self.conv2 is not None:
            x = self.conv2(x)

        x1, x2 = x.chunk(2, dim=1)

        x1 = self.branch1(x1)

        x2 = self.expand_conv(x2)
        x2 = self.depthwise_conv(x2)
        x2 = self.linear_conv(x2)

        out = torch.cat((x1, x2), dim=1)

        return channel_shuffle(out, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return cp.checkpoint(self._inner_forward, x) if self.with_cp and x.requires_grad else self._inner_forward(x)


class StemV2(nn.Module):
    """StemV2.

    Args:
        in_channels (int): Number of input image channels. Typically 3.
        stem_channels (int): Number of output channels of the stem layer.
        out_channels (int): Number of output channels of the backbone network.
        expand_ratio (int): Expansion ratio of the internal channels.
        normalization_callable (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``nn.BatchNorm2d``.
        with_cp (bool): Use checkpointing to save memory during forward pass.
        num_stages (int): Number of stages in the backbone network.
        strides (tuple[int, int]): Strides of the first and subsequent stages.
        extra_stride (bool): Use an extra stride in the second stage.
        input_norm (bool): Use instance normalization on the input image.

    Raises:
        ValueError: If num_stages is less than 1.
        TypeError: If strides is not a tuple or list.
        ValueError: If len(strides) is not equal to num_stages + 1.
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        out_channels: int,
        expand_ratio: int,
        normalization_callable: Callable[..., nn.Module] = nn.BatchNorm2d,
        with_cp: bool = False,
        num_stages: int = 1,
        strides: tuple[int, int] = (2, 2),
        extra_stride: bool = False,
        input_norm: bool = False,
    ) -> None:
        """StemV2 initialization."""
        super().__init__()

        if num_stages < 1:
            msg = "num_stages must be greater than 0."
            raise ValueError(msg)
        if not isinstance(strides, (tuple, list)):
            msg = "strides must be tuple or list."
            raise TypeError(msg)

        if len(strides) != 1 + num_stages:
            msg = "len(strides) must equal to num_stages + 1."
            raise ValueError(msg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization_callable = normalization_callable
        self.with_cp = with_cp
        self.num_stages = num_stages

        self.input_norm = None
        if input_norm:
            self.input_norm = nn.InstanceNorm2d(in_channels)

        self.conv1 = Conv2dModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=strides[0],
            padding=1,
            normalization=build_norm_layer(self.normalization_callable, num_features=stem_channels),
            activation_callable=nn.ReLU,
        )

        self.conv2 = None
        if extra_stride:
            self.conv2 = Conv2dModule(
                in_channels=stem_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                normalization=build_norm_layer(self.normalization_callable, num_features=stem_channels),
                activation_callable=nn.ReLU,
            )

        mid_channels = int(round(stem_channels * expand_ratio))
        internal_branch_channels = stem_channels // 2
        out_branch_channels = self.out_channels // 2

        self.branch1, self.branch2 = nn.ModuleList(), nn.ModuleList()
        for stage in range(1, num_stages + 1):
            self.branch1.append(
                nn.Sequential(
                    Conv2dModule(
                        internal_branch_channels,
                        internal_branch_channels,
                        kernel_size=3,
                        stride=strides[stage],
                        padding=1,
                        groups=internal_branch_channels,
                        normalization=build_norm_layer(normalization_callable, num_features=internal_branch_channels),
                        activation_callable=None,
                    ),
                    Conv2dModule(
                        internal_branch_channels,
                        out_branch_channels if stage == num_stages else internal_branch_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        normalization=build_norm_layer(
                            normalization_callable,
                            num_features=out_branch_channels if stage == num_stages else internal_branch_channels,
                        ),
                        activation_callable=nn.ReLU,
                    ),
                ),
            )

            self.branch2.append(
                nn.Sequential(
                    Conv2dModule(
                        internal_branch_channels,
                        mid_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        normalization=build_norm_layer(normalization_callable, num_features=mid_channels),
                        activation_callable=nn.ReLU,
                    ),
                    Conv2dModule(
                        mid_channels,
                        mid_channels,
                        kernel_size=3,
                        stride=strides[stage],
                        padding=1,
                        groups=mid_channels,
                        normalization=build_norm_layer(normalization_callable, num_features=mid_channels),
                        activation_callable=None,
                    ),
                    Conv2dModule(
                        mid_channels,
                        out_branch_channels if stage == num_stages else internal_branch_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        normalization=build_norm_layer(
                            normalization_callable,
                            num_features=out_branch_channels if stage == num_stages else internal_branch_channels,
                        ),
                        activation_callable=nn.ReLU,
                    ),
                ),
            )

    def _inner_forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass of Stem module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            list[torch.Tensor]: List of output tensors at each stage of the backbone.
        """
        if self.input_norm is not None:
            x = self.input_norm(x)

        y = self.conv1(x)
        if self.conv2 is not None:
            y = self.conv2(y)

        out_list = [y]
        for stage in range(self.num_stages):
            y1, y2 = y.chunk(2, dim=1)

            y1 = self.branch1[stage](y1)
            y2 = self.branch2[stage](y2)

            y = torch.cat((y1, y2), dim=1)
            y = channel_shuffle(y, 2)
            out_list.append(y)

        return out_list

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward."""
        return cp.checkpoint(self._inner_forward, x) if self.with_cp and x.requires_grad else self._inner_forward(x)


class ShuffleUnit(nn.Module):
    """InvertedResidual block for ShuffleNetV2 backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        normalization_callable (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``nn.BatchNorm2d``.
        activation_callable (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.ReLU``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        normalization_callable: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_callable: Callable[..., nn.Module] = nn.ReLU,
        with_cp: bool = False,
    ) -> None:
        super().__init__()

        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        if self.stride == 1 and in_channels != branch_features * 2:
            msg = (
                f"in_channels ({in_channels}) should equal to "
                f"branch_features * 2 ({branch_features * 2}) "
                "when stride is 1"
            )
            raise ValueError(msg)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                Conv2dModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels,
                    normalization=build_norm_layer(normalization_callable, num_features=in_channels),
                    activation_callable=None,
                ),
                Conv2dModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    normalization=build_norm_layer(normalization_callable, num_features=branch_features),
                    activation_callable=activation_callable,
                ),
            )

        self.branch2 = nn.Sequential(
            Conv2dModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                normalization=build_norm_layer(normalization_callable, num_features=branch_features),
                activation_callable=activation_callable,
            ),
            Conv2dModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features,
                normalization=build_norm_layer(normalization_callable, num_features=branch_features),
                activation_callable=None,
            ),
            Conv2dModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                normalization=build_norm_layer(normalization_callable, num_features=branch_features),
                activation_callable=activation_callable,
            ),
        )

    def _inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """_inner_forward."""
        if self.stride > 1:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)

        return channel_shuffle(out, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return cp.checkpoint(self._inner_forward, x) if self.with_cp and x.requires_grad else self._inner_forward(x)


class LiteHRModule(nn.Module):
    """LiteHR module.

    Args:
        num_branches (int): Number of branches in the network.
        num_blocks (int): Number of blocks in each branch.
        in_channels (list[int]): List of input channels for each branch.
        reduce_ratio (int): Reduction ratio for the weighting module.
        module_type (str): Type of module to use for the network. Can be "LITE" or "NAIVE".
        multiscale_output (bool, optional): Whether to output features from all branches. Defaults to False.
        with_fuse (bool, optional): Whether to use the fuse layer. Defaults to True.
        normalization_callable (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``nn.BatchNorm2d``.
        with_cp (bool, optional): Whether to use checkpointing. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to None.
        weighting_module_version (str, optional): Version of the weighting module to use. Defaults to "v1".
        neighbour_weighting (bool, optional): Whether to use neighbour weighting. Defaults to False.
    """

    def __init__(
        self,
        num_branches: int,
        num_blocks: int,
        in_channels: list[int],
        reduce_ratio: int,
        module_type: str,
        multiscale_output: bool = False,
        with_fuse: bool = True,
        normalization_callable: Callable[..., nn.Module] = nn.BatchNorm2d,
        with_cp: bool = False,
        dropout: float | None = None,
        weighting_module_version: str = "v1",
        neighbour_weighting: bool = False,
    ) -> None:
        super().__init__()

        self._check_branches(num_branches, in_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.normalization_callable = normalization_callable
        self.with_cp = with_cp
        self.weighting_module_version = weighting_module_version
        self.neighbour_weighting = neighbour_weighting

        if self.module_type == "LITE":
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio, dropout=dropout)
        elif self.module_type == "NAIVE":
            self.layers = self._make_naive_branches(num_branches, num_blocks)

        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    @staticmethod
    def _check_branches(num_branches: int, in_channels: list[int]) -> None:
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = f"NUM_BRANCHES({num_branches}) != NUM_INCHANNELS({len(in_channels)})"
            raise ValueError(error_msg)

    def _make_weighting_blocks(
        self,
        num_blocks: int,
        reduce_ratio: int,
        stride: int = 1,
        dropout: float | None = None,
    ) -> nn.Sequential:
        layers = [
            ConditionalChannelWeighting(
                self.in_channels,
                stride=stride,
                reduce_ratio=reduce_ratio,
                normalization_callable=self.normalization_callable,
                with_cp=self.with_cp,
                dropout=dropout,
                weighting_module_version=self.weighting_module_version,
                neighbour_weighting=self.neighbour_weighting,
            )
            for _ in range(num_blocks)
        ]

        return nn.Sequential(*layers)

    def _make_one_branch(self, branch_index: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """Make one branch."""
        layers = [
            ShuffleUnit(
                self.in_channels[branch_index],
                self.in_channels[branch_index],
                stride=stride,
                normalization_callable=self.normalization_callable,
                activation_callable=nn.ReLU,
                with_cp=self.with_cp,
            ),
        ] + [
            ShuffleUnit(
                self.in_channels[branch_index],
                self.in_channels[branch_index],
                stride=1,
                normalization_callable=self.normalization_callable,
                activation_callable=nn.ReLU,
                with_cp=self.with_cp,
            )
            for _ in range(1, num_blocks)
        ]

        return nn.Sequential(*layers)

    def _make_naive_branches(self, num_branches: int, num_blocks: int) -> nn.ModuleList:
        """Make branches."""
        branches = [self._make_one_branch(i, num_blocks) for i in range(num_branches)]
        return nn.ModuleList(branches)

    def _make_fuse_layers(self) -> nn.ModuleList:
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        num_out_branches = num_branches if self.multiscale_output else 1

        fuse_layers = []
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.normalization_callable, in_channels[i])[1],
                        ),
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False,
                                    ),
                                    build_norm_layer(self.normalization_callable, in_channels[j])[1],
                                    nn.Conv2d(
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.normalization_callable, in_channels[i])[1],
                                ),
                            )
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False,
                                    ),
                                    build_norm_layer(self.normalization_callable, in_channels[j])[1],
                                    nn.Conv2d(
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.normalization_callable, in_channels[j])[1],
                                    nn.ReLU(inplace=True),
                                ),
                            )
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward function."""
        if self.num_branches == 1:
            return [self.layers[0](x[0])]

        if self.module_type == "LITE":
            out = self.layers(x)
        elif self.module_type == "NAIVE":
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            out = x

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    fuse_y = out[j] if i == j else self.fuse_layers[i][j](out[j])
                    if fuse_y.size()[-2:] != y.size()[-2:]:
                        fuse_y = functional.interpolate(fuse_y, size=y.size()[-2:], mode="nearest")

                    y += fuse_y

                out_fuse.append(self.relu(y))

            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]

        return out


class LiteHRNet(BaseModule):
    """Lite-HRNet backbone.

    `High-Resolution Representations for Labeling Pixels and Regions
    <https://arxiv.org/abs/1904.04514>`_

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Default: 3.
        normalization_callable (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``nn.BatchNorm2d``.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    def __init__(
        self,
        extra: dict,
        in_channels: int = 3,
        normalization_callable: Callable[..., nn.Module] = nn.BatchNorm2d,
        norm_eval: bool = False,
        with_cp: bool = False,
        zero_init_residual: bool = False,
        dropout: float | None = None,
        init_cfg: dict | None = None,
        pretrained_weights: str | None = None,
    ) -> None:
        """Init."""
        super().__init__(init_cfg=init_cfg)

        self.extra = extra
        self.normalization_callable = normalization_callable
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.stem = Stem(
            in_channels,
            input_norm=self.extra["stem"]["input_norm"],
            stem_channels=self.extra["stem"]["stem_channels"],
            out_channels=self.extra["stem"]["out_channels"],
            expand_ratio=self.extra["stem"]["expand_ratio"],
            strides=self.extra["stem"]["strides"],
            extra_stride=self.extra["stem"]["extra_stride"],
            normalization_callable=self.normalization_callable,
        )

        self.enable_stem_pool = self.extra["stem"].get("out_pool", False)
        if self.enable_stem_pool:
            self.stem_pool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.num_stages = self.extra["num_stages"]
        self.stages_spec = self.extra["stages_spec"]

        num_channels_last = [
            self.stem.out_channels,
        ]
        for i in range(self.num_stages):
            num_channels = self.stages_spec["num_channels"][i]
            num_channels = [num_channels[i] for i in range(len(num_channels))]

            setattr(
                self,
                f"transition{i}",
                self._make_transition_layer(num_channels_last, num_channels),
            )

            stage, num_channels_last = self._make_stage(
                self.stages_spec,
                i,
                num_channels,
                multiscale_output=True,
                dropout=dropout,
            )
            setattr(self, f"stage{i}", stage)

        self.out_modules = None
        if self.extra.get("out_modules") is not None:
            out_modules = []
            in_modules_channels, out_modules_channels = num_channels_last[-1], None
            if self.extra["out_modules"]["conv"]["enable"]:
                out_modules_channels = self.extra["out_modules"]["conv"]["channels"]
                out_modules.append(
                    Conv2dModule(
                        in_channels=in_modules_channels,
                        out_channels=out_modules_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        normalization=build_norm_layer(self.normalization_callable, num_features=out_modules_channels),
                        activation_callable=nn.ReLU,
                    ),
                )
                in_modules_channels = out_modules_channels
            if self.extra["out_modules"]["position_att"]["enable"]:
                out_modules.append(
                    AsymmetricPositionAttentionModule(
                        in_channels=in_modules_channels,
                        key_channels=self.extra["out_modules"]["position_att"]["key_channels"],
                        value_channels=self.extra["out_modules"]["position_att"]["value_channels"],
                        psp_size=self.extra["out_modules"]["position_att"]["psp_size"],
                        normalization_callable=self.normalization_callable,
                    ),
                )
            if self.extra["out_modules"]["local_att"]["enable"]:
                out_modules.append(
                    LocalAttentionModule(
                        num_channels=in_modules_channels,
                        normalization_callable=self.normalization_callable,
                    ),
                )

            if len(out_modules) > 0:
                self.out_modules = nn.Sequential(*out_modules)
                num_channels_last.append(in_modules_channels)

        self.add_stem_features = self.extra.get("add_stem_features", False)
        if self.add_stem_features:
            self.stem_transition = nn.Sequential(
                Conv2dModule(
                    self.stem.out_channels,
                    self.stem.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=self.stem.out_channels,
                    normalization=build_norm_layer(normalization_callable, num_features=self.stem.out_channels),
                    activation_callable=None,
                ),
                Conv2dModule(
                    self.stem.out_channels,
                    num_channels_last[0],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    normalization=build_norm_layer(normalization_callable, num_features=num_channels_last[0]),
                    activation_callable=nn.ReLU,
                ),
            )

            num_channels_last = [num_channels_last[0], *num_channels_last]

        self.with_aggregator = self.extra.get("out_aggregator") and self.extra["out_aggregator"]["enable"]
        if self.with_aggregator:
            self.aggregator = IterativeAggregator(
                in_channels=num_channels_last,
                min_channels=self.extra["out_aggregator"].get("min_channels", None),
                normalization_callable=self.normalization_callable,
            )

        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights, prefix="backbone")

    def _make_transition_layer(
        self,
        num_channels_pre_layer: list[int],
        num_channels_cur_layer: list[int],
    ) -> nn.ModuleList:
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False,
                            ),
                            build_norm_layer(self.normalization_callable, num_channels_pre_layer[i])[1],
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.normalization_callable, num_channels_cur_layer[i])[1],
                            nn.ReLU(),
                        ),
                    )
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=in_channels,
                                bias=False,
                            ),
                            build_norm_layer(self.normalization_callable, in_channels)[1],
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.normalization_callable, out_channels)[1],
                            nn.ReLU(),
                        ),
                    )
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        stages_spec: dict,
        stage_index: int,
        in_channels: list[int],
        multiscale_output: bool = True,
        dropout: float | None = None,
    ) -> tuple[nn.Module, list[int]]:
        """Create a stage of the LiteHRNet backbone.

        Args:
            stages_spec (dict): Specification of the stages of the backbone.
            stage_index (int): Index of the current stage.
            in_channels (list[int]): List of input channels for each branch.
            multiscale_output (bool, optional): Whether to output features from all branches. Defaults to True.
            dropout (float | None, optional): Dropout probability. Defaults to None.

        Returns:
            tuple[nn.Module, list[int]]: A tuple containing the stage module and the output channels for each branch.
        """
        num_modules = stages_spec["num_modules"][stage_index]
        num_branches = stages_spec["num_branches"][stage_index]
        num_blocks = stages_spec["num_blocks"][stage_index]
        reduce_ratio = stages_spec["reduce_ratios"][stage_index]
        with_fuse = stages_spec["with_fuse"][stage_index]
        module_type = stages_spec["module_type"][stage_index]
        weighting_module_version = stages_spec.get("weighting_module_version", "v1")
        neighbour_weighting = stages_spec.get("neighbour_weighting", False)

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            reset_multiscale_output = not ((not multiscale_output) and i == num_modules - 1)

            modules.append(
                LiteHRModule(
                    num_branches,
                    num_blocks,
                    in_channels,
                    reduce_ratio,
                    module_type,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=with_fuse,
                    normalization_callable=self.normalization_callable,
                    with_cp=self.with_cp,
                    dropout=dropout,
                    weighting_module_version=weighting_module_version,
                    neighbour_weighting=neighbour_weighting,
                ),
            )
            in_channels = modules[-1].in_channels

        return nn.Sequential(*modules), in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        stem_outputs = self.stem(x)
        y_x2 = y_x4 = stem_outputs
        y = y_x4

        if self.enable_stem_pool:
            y = self.stem_pool(y)

        y_list = [y]
        for i in range(self.num_stages):
            transition_modules = getattr(self, f"transition{i}")

            stage_inputs = []
            for j in range(self.stages_spec["num_branches"][i]):
                if transition_modules[j]:
                    if j >= len(y_list):
                        stage_inputs.append(transition_modules[j](y_list[-1]))
                    else:
                        stage_inputs.append(transition_modules[j](y_list[j]))
                else:
                    stage_inputs.append(y_list[j])

            stage_module = getattr(self, f"stage{i}")
            y_list = stage_module(stage_inputs)

        if self.out_modules is not None:
            y_list.append(self.out_modules(y_list[-1]))

        if self.add_stem_features:
            y_stem = self.stem_transition(y_x2)
            y_list = [y_stem, *y_list]

        out = y_list
        if self.with_aggregator:
            out = self.aggregator(out)

        if self.extra.get("add_input", False):
            out = [x, *out]

        return out

    def load_pretrained_weights(self, pretrained: str | None = None, prefix: str = "") -> None:
        """Initialize weights."""
        checkpoint = None
        if isinstance(pretrained, str) and Path(pretrained).exists():
            checkpoint = torch.load(pretrained, "cpu")
            print(f"init weight - {pretrained}")
        elif pretrained is not None:
            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            checkpoint = load_from_http(filename=pretrained, map_location="cpu", model_dir=cache_dir)
            print(f"init weight - {pretrained}")
        if checkpoint is not None:
            load_checkpoint_to_model(self, checkpoint, prefix=prefix)
