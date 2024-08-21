# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet ResLayer."""

from __future__ import annotations

from typing import Callable

from otx.algo.modules.base_module import BaseModule, Sequential
from otx.algo.modules.norm import build_norm_layer
from torch import nn


class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Defaults to 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``nn.BatchNorm2d``.
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Defaults to True
    """

    def __init__(
        self,
        block: BaseModule,
        inplanes: int,
        planes: int,
        num_blocks: int,
        normalization: Callable[..., nn.Module],
        stride: int = 1,
        avg_down: bool = False,
        downsample_first: bool = True,
        **kwargs,
    ) -> None:
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False,
                    ),
                )
            downsample.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias=False,
                    ),
                    build_norm_layer(normalization, planes * block.expansion)[1],
                ],
            )
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    normalization=normalization,
                    **kwargs,
                ),
            )
            inplanes = planes * block.expansion
            layers.extend(
                [
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        normalization=normalization,
                        **kwargs,
                    )
                    for _ in range(1, num_blocks)
                ],
            )

        super().__init__(*layers)
