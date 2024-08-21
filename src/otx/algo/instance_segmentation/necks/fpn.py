# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.necks.fpn.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/necks/fpn.py
"""

from __future__ import annotations

from typing import Callable

import torch.nn.functional
from torch import Tensor, nn

from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.conv_module import Conv2dModule
from otx.algo.modules.norm import build_norm_layer


class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to None.
        activation (Callable[..., nn.Module] | None): Activation layer module.
            Defaults to None.
        upsample_cfg (dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (dict or list[dict]): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        normalization: Callable[..., nn.Module] | None = None,
        activation: Callable[..., nn.Module] | None = None,
        upsample_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        init_cfg = {"type": "Xavier", "layer": "Conv2d", "distribution": "uniform"} if init_cfg is None else init_cfg
        super().__init__(init_cfg=init_cfg)
        if not isinstance(in_channels, list):
            msg = f"in_channels must be a list, but got {type(in_channels)}"
            raise AssertionError(msg)  # noqa: TRY004
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = {"mode": "nearest"} if upsample_cfg is None else upsample_cfg

        if end_level in (-1, self.num_ins - 1):
            self.backbone_end_level = self.num_ins
            if num_outs < self.num_ins - start_level:
                msg = "num_outs should not be less than the number of output levels"
                raise ValueError(msg)
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            if end_level >= self.num_ins:
                msg = "end_level must be less than len(in_channels)"
                raise ValueError(msg)
            if num_outs != end_level - start_level + 1:
                msg = "num_outs must be equal to end_level - start_level + 1"
                raise ValueError(msg)
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = Conv2dModule(
                in_channels[i],
                out_channels,
                1,
                normalization=build_norm_layer(normalization, num_features=out_channels)
                if not self.no_norm_on_lateral
                else None,
                activation=build_activation_layer(activation),
                inplace=False,
            )
            fpn_conv = Conv2dModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalization=build_norm_layer(normalization, num_features=out_channels),
                activation=build_activation_layer(activation),
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs: tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        if len(inputs) != len(self.in_channels):
            msg = f"len(inputs) is not equal to len(in_channels): {len(inputs)} != {len(self.in_channels)}"
            raise ValueError(msg)

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if "scale_factor" in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + torch.nn.functional.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + torch.nn.functional.interpolate(
                    laterals[i],
                    size=prev_shape,
                    **self.upsample_cfg,
                )

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            for _ in range(self.num_outs - used_backbone_levels):
                outs.append(torch.nn.functional.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)
