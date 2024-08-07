# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom FCNHead modules for OTX segmentation model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from otx.algo.modules import Conv2dModule
from otx.algo.segmentation.modules import IterativeAggregator

from .base_segm_head import BaseSegmHead


class FCNHead(BaseSegmHead):
    """Fully Convolution Networks for Semantic Segmentation with aggregation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self,
        in_channels: list[int] | int,
        in_index: list[int] | int,
        norm_cfg: dict[str, Any] | None = None,
        input_transform: str | None = None,
        num_convs: int = 2,
        kernel_size: int = 3,
        concat_input: bool = True,
        dilation: int = 1,
        enable_aggregator: bool = False,
        aggregator_min_channels: int = 0,
        aggregator_merge_norm: str | None = None,
        aggregator_use_concat: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize a Fully Convolution Networks head.

        Args:
            num_convs (int): Number of convs in the head.
            kernel_size (int): The kernel size for convs in the head.
            concat_input (bool): Whether to concat input and output of convs.
            dilation (int): The dilation rate for convs in the head.
            **kwargs: Additional arguments.
        """
        if not isinstance(dilation, int):
            msg = f"dilation should be int, but got {type(dilation)}"
            raise TypeError(msg)
        if num_convs < 0 and dilation <= 0:
            msg = "num_convs and dilation should be larger than 0"
            raise ValueError(msg)

        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size

        if enable_aggregator:  # Lite-HRNet aggregator
            if in_channels is None or isinstance(in_channels, int):
                msg = "'in_channels' should be List[int]."
                raise ValueError(msg)
            aggregator = IterativeAggregator(
                in_channels=in_channels,
                min_channels=aggregator_min_channels,
                norm_cfg=norm_cfg,
                merge_norm=aggregator_merge_norm,
                use_concat=aggregator_use_concat,
            )

            aggregator_min_channels = aggregator_min_channels if aggregator_min_channels is not None else 0
            # change arguments temporarily
            in_channels = max(in_channels[0], aggregator_min_channels)
            input_transform = None
            if isinstance(in_index, list):
                in_index = in_index[0]
        else:
            aggregator = None

        super().__init__(
            in_index=in_index,
            norm_cfg=norm_cfg,
            input_transform=input_transform,
            in_channels=in_channels,
            **kwargs,
        )

        self.aggregator = aggregator

        if num_convs == 0 and (self.in_channels != self.channels):
            msg = "in_channels and channels should be equal when num_convs is 0"
            raise ValueError(msg)

        conv_padding = (kernel_size // 2) * dilation
        convs = [
            Conv2dModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        ]
        convs.extend(
            [
                Conv2dModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
                for _ in range(num_convs - 1)
            ],
        )
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = Conv2dModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )

        if self.act_cfg:
            self.convs[-1].with_activation = False
            delattr(self.convs[-1], "activate")  # why we delete last activation?

    def _forward_feature(self, inputs: Tensor) -> Tensor:
        """Forward function for feature maps.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward function."""
        output = self._forward_feature(inputs)
        return self.cls_seg(output)

    def _transform_inputs(self, inputs: list[Tensor]) -> Tensor | list:
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        return self.aggregator(inputs)[0] if self.aggregator is not None else super()._transform_inputs(inputs)
