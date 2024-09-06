# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""ELAN layers implementation for YOLOv7 and v9.

Reference : https://github.com/WongKinYiu/YOLO
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from torch import Tensor, nn

from otx.algo.detection.utils.utils import auto_pad
from otx.algo.modules import Conv2dModule, build_activation_layer

logger = logging.getLogger(__name__)


class ELAN(nn.Module):
    """ELAN structure.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        part_channels (int): The number of part channels.
        process_channels (int | None, optional): The number of process channels. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv2dModule(
            in_channels,
            part_channels,
            1,
            normalization=nn.BatchNorm2d(part_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv2 = Conv2dModule(
            part_channels // 2,
            process_channels,
            3,
            padding=1,
            normalization=nn.BatchNorm2d(process_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv3 = Conv2dModule(
            process_channels,
            process_channels,
            3,
            padding=1,
            normalization=nn.BatchNorm2d(process_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv4 = Conv2dModule(
            part_channels + 2 * process_channels,
            out_channels,
            1,
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for ELAN."""
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return self.conv4(torch.cat([x1, x2, x3, x4], dim=1))


class RepConv(nn.Module):
    """A convolutional block that combines two convolution layers (kernel and point-wise).

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (tuple[int, int], optional): The kernel size. Defaults to 3.
        activation (Callable[..., nn.Module], optional): The activation function. Defaults to
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        *,
        activation: Callable[..., nn.Module] = nn.SiLU,
        **kwargs,
    ) -> None:
        super().__init__()
        self.act: nn.Module = build_activation_layer(activation)
        self.conv1 = Conv2dModule(
            in_channels,
            out_channels,
            kernel_size,
            padding=auto_pad(kernel_size=kernel_size, **kwargs),
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=None,
            **kwargs,
        )
        self.conv2 = Conv2dModule(
            in_channels,
            out_channels,
            1,
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=None,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for RepConv."""
        return self.act(self.conv1(x) + self.conv2(x))


class RepNCSPBottleneck(nn.Module):
    """A bottleneck block with optional residual connections.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (tuple[int, int], optional): The kernel size. Defaults to (3, 3).
        residual (bool, optional): Whether to use residual connections. Defaults to True.
        expand (float, optional): The expansion factor. Defaults to 1.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int] = (3, 3),
        residual: bool = True,
        expand: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        neck_channels = int(out_channels * expand)
        self.conv1 = RepConv(in_channels, neck_channels, kernel_size[0], **kwargs)
        self.conv2 = Conv2dModule(
            neck_channels,
            out_channels,
            kernel_size[1],
            padding=auto_pad(kernel_size=kernel_size, **kwargs),
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.residual = residual

        if residual and (in_channels != out_channels):
            self.residual = False
            msg = f"Residual connection disabled: in_channels ({in_channels}) != out_channels ({out_channels})"
            logger.warning(msg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for Bottleneck."""
        y = self.conv2(self.conv1(x))
        return x + y if self.residual else y


class RepNCSP(nn.Module):
    """RepNCSP block with convolutions, split, and bottleneck processing.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int, optional): The kernel size. Defaults to 1.
        csp_expand (float, optional): The expansion factor for CSP. Defaults to 0.5.
        repeat_num (int, optional): The number of repetitions. Defaults to 1.
        neck_args (dict[str, Any] | None, optional): The configuration for the bottleneck blocks. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        csp_expand: float = 0.5,
        repeat_num: int = 1,
        neck_args: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        neck_args = neck_args or {}
        neck_channels = int(out_channels * csp_expand)
        self.conv1 = Conv2dModule(
            in_channels,
            neck_channels,
            kernel_size,
            padding=auto_pad(kernel_size=kernel_size, **kwargs),
            normalization=nn.BatchNorm2d(neck_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv2 = Conv2dModule(
            in_channels,
            neck_channels,
            kernel_size,
            padding=auto_pad(kernel_size=kernel_size, **kwargs),
            normalization=nn.BatchNorm2d(neck_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv3 = Conv2dModule(
            2 * neck_channels,
            out_channels,
            kernel_size,
            padding=auto_pad(kernel_size=kernel_size, **kwargs),
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )

        self.bottleneck = nn.Sequential(
            *[RepNCSPBottleneck(neck_channels, neck_channels, **neck_args) for _ in range(repeat_num)],
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for RepNCSP."""
        x1 = self.bottleneck(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))


class RepNCSPELAN(nn.Module):
    """RepNCSPELAN block combining RepNCSP blocks with ELAN structure.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        part_channels (int): The number of part channels.
        process_channels (int | None, optional): The number of process channels. Defaults to None.
        csp_args (dict[str, Any] | None, optional): The configuration for the CSP blocks. Defaults to None.
        csp_neck_args (dict[str, Any] | None, optional): The configuration for the CSP neck blocks. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: int | None = None,
        csp_args: dict[str, Any] | None = None,
        csp_neck_args: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        csp_args = csp_args or {}
        csp_neck_args = csp_neck_args or {}
        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv2dModule(
            in_channels,
            part_channels,
            1,
            normalization=nn.BatchNorm2d(part_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv2 = nn.Sequential(
            RepNCSP(part_channels // 2, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv2dModule(
                process_channels,
                process_channels,
                3,
                padding=1,
                normalization=nn.BatchNorm2d(process_channels, eps=1e-3, momentum=3e-2),
                activation=nn.SiLU(inplace=True),
                **kwargs,
            ),
        )
        self.conv3 = nn.Sequential(
            RepNCSP(process_channels, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv2dModule(
                process_channels,
                process_channels,
                3,
                padding=1,
                normalization=nn.BatchNorm2d(process_channels, eps=1e-3, momentum=3e-2),
                activation=nn.SiLU(inplace=True),
                **kwargs,
            ),
        )
        self.conv4 = Conv2dModule(
            part_channels + 2 * process_channels,
            out_channels,
            1,
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for RepNCSPELAN."""
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return self.conv4(torch.cat([x1, x2, x3, x4], dim=1))


class SPPELAN(nn.Module):
    """SPPELAN module comprising multiple pooling and convolution layers.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        neck_channels (int | None): The number of neck channels. Defaults to None.
    """

    def __init__(self, in_channels: int, out_channels: int, neck_channels: int | None = None) -> None:
        super().__init__()
        neck_channels = neck_channels or out_channels // 2

        self.conv1 = Conv2dModule(
            in_channels,
            neck_channels,
            kernel_size=1,
            normalization=nn.BatchNorm2d(neck_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
        )
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=5, stride=1, padding=auto_pad(kernel_size=5)) for _ in range(3)],
        )
        self.conv5 = Conv2dModule(
            4 * neck_channels,
            out_channels,
            kernel_size=1,
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        features = [self.conv1(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        return self.conv5(torch.cat(features, dim=1))
