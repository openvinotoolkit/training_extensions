"""The original source code is from mmdet. Please refer to https://github.com/open-mmlab/mmdetection/."""

# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

# TODO(Eugene): replace mmcv with torchvision
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule, Sequential
from torch import nn

from otx.algo.instance_segmentation.mmdet.models.utils import OptConfigType


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
        conv_cfg (dict): dictionary to construct and config conv layer.
            Defaults to None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Defaults to dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Defaults to True
    """

    def __init__(
        self,
        block: BaseModule,
        inplanes: int,
        planes: int,
        num_blocks: int,
        norm_cfg: dict,
        stride: int = 1,
        avg_down: bool = False,
        conv_cfg: OptConfigType = None,
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
                    build_conv_layer(
                        conv_cfg,
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias=False,
                    ),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1],
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
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs,
                ),
            )
            inplanes = planes * block.expansion
            layers.extend(
                [
                    block(inplanes=inplanes, planes=planes, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs)
                    for _ in range(1, num_blocks)
                ],
            )

        super().__init__(*layers)
