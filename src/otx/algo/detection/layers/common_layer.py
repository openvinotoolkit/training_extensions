# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Common layers implementation.

Reference : https://github.com/WongKinYiu/YOLO
"""

import torch
from torch import Tensor, nn

from otx.algo.detection.utils.utils import auto_pad
from otx.algo.modules import Conv2dModule


class Concat(nn.Module):
    """Concat module.

    Args:
        dim (int): The dimension to concatenate. Defaults to 1.
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        return torch.cat(x, self.dim)


class AConv(nn.Module):
    """Downsampling module combining average and max pooling with convolution for feature reduction.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=auto_pad(kernel_size=2))
        self.conv = Conv2dModule(
            in_channels,
            out_channels,
            3,
            stride=2,
            padding=auto_pad(kernel_size=3),
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for AConv."""
        x = self.avg_pool(x)
        return self.conv(x)


class ADown(nn.Module):
    """Downsampling module combining average and max pooling with convolution for feature reduction.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        half_in_channels = in_channels // 2
        half_out_channels = out_channels // 2
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=auto_pad(kernel_size=2))
        self.conv1 = Conv2dModule(
            half_in_channels,
            half_out_channels,
            3,
            stride=2,
            padding=auto_pad(kernel_size=3),
            normalization=nn.BatchNorm2d(half_out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=auto_pad(kernel_size=3))
        self.conv2 = Conv2dModule(
            half_in_channels,
            half_out_channels,
            1,
            normalization=nn.BatchNorm2d(half_out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for ADown."""
        x = self.avg_pool(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.max_pool(x2)
        x2 = self.conv2(x2)
        return torch.cat((x1, x2), dim=1)
