# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""FPN (Feature Pyramid Network) implementation.

Implementation modified from mmdet.models.necks.fpn.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/necks/fpn.py
"""

from __future__ import annotations

from typing import Callable

from torch import Tensor, nn

from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.conv_module import Conv2dModule
from otx.algo.modules.norm import build_norm_layer


class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
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
        init_cfg (dict, list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: bool | str = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        normalization: Callable[..., nn.Module] | None = None,
        activation: Callable[..., nn.Module] | None = None,
        upsample_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        init_cfg = init_cfg or {"type": "Xavier", "layer": "Conv2d", "distribution": "uniform"}
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy() if upsample_cfg is not None else {"mode": "nearest"}

        if end_level in (-1, self.num_ins - 1):
            self.backbone_end_level = self.num_ins
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            if add_extra_convs not in ("on_input", "on_lateral", "on_output"):
                msg = f'add_extra_convs: {add_extra_convs} not in ("on_input", "on_lateral", "on_output")'
                raise ValueError(msg)
        elif add_extra_convs:  # True
            self.add_extra_convs = "on_input"

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

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    conv_in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    conv_in_channels = out_channels
                extra_fpn_conv = Conv2dModule(
                    conv_in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalization=build_norm_layer(normalization, num_features=out_channels),
                    activation=build_activation_layer(activation),
                    inplace=False,
                )
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs: tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if "scale_factor" in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
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
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(nn.functional.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == "on_input":
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                elif self.add_extra_convs == "on_output":
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](nn.functional.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
