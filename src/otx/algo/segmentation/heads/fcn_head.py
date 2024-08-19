# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom FCNHead modules for OTX segmentation model."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch
from torch import Tensor, nn

from otx.algo.modules import Conv2dModule
from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.norm import build_norm_layer
from otx.algo.segmentation.modules import IterativeAggregator

from .base_segm_head import BaseSegmHead

if TYPE_CHECKING:
    from pathlib import Path


class NNFCNHead(BaseSegmHead):
    """Fully Convolution Networks for Semantic Segmentation with aggregation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to None.
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
        channels: int,
        normalization: Callable[..., nn.Module] = partial(build_norm_layer, nn.BatchNorm2d, requires_grad=True),
        input_transform: str | None = None,
        num_classes: int = 80,
        num_convs: int = 1,
        kernel_size: int = 1,
        concat_input: bool = False,
        dilation: int = 1,
        enable_aggregator: bool = False,
        aggregator_min_channels: int = 0,
        aggregator_merge_norm: str | None = None,
        aggregator_use_concat: bool = False,
        align_corners: bool = False,
        dropout_ratio: float = -1,
        activation: Callable[..., nn.Module] | None = nn.ReLU,
        pretrained_weights: Path | str | None = None,
    ) -> None:
        """Initialize a Fully Convolution Networks head."""
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
                normalization=normalization,
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
            normalization=normalization,
            input_transform=input_transform,
            in_channels=in_channels,
            align_corners=align_corners,
            dropout_ratio=dropout_ratio,
            channels=channels,
            num_classes=num_classes,
            activation=activation,
            pretrained_weights=pretrained_weights,
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
                normalization=build_norm_layer(self.normalization, num_features=self.channels),
                activation=build_activation_layer(self.activation),
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
                    normalization=build_norm_layer(self.normalization, num_features=self.channels),
                    activation=build_activation_layer(self.activation),
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
                normalization=build_norm_layer(self.normalization, num_features=self.channels),
                activation=build_activation_layer(self.activation),
            )

        if self.activation:
            self.convs[-1].with_activation = False
            delattr(self.convs[-1], "activation")  # why we delete last activation?

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


class FCNHead:
    """FCNHead factory for segmentation."""

    FCNHEAD_CFG: ClassVar[dict[str, Any]] = {
        "lite_hrnet_s": {
            "in_channels": [60, 120, 240],
            "in_index": [0, 1, 2],
            "input_transform": "multiple_select",
            "channels": 60,
            "enable_aggregator": True,
            "aggregator_merge_norm": "None",
            "aggregator_use_concat": False,
        },
        "lite_hrnet_18": {
            "in_channels": [40, 80, 160, 320],
            "in_index": [0, 1, 2, 3],
            "input_transform": "multiple_select",
            "channels": 40,
            "enable_aggregator": True,
        },
        "lite_hrnet_x": {
            "in_channels": [18, 60, 80, 160, 320],
            "in_index": [0, 1, 2, 3, 4],
            "input_transform": "multiple_select",
            "channels": 60,
            "enable_aggregator": True,
            "aggregator_min_channels": 60,
            "aggregator_merge_norm": "None",
            "aggregator_use_concat": False,
        },
        "dinov2_vits14": {
            "normalization": partial(build_norm_layer, nn.SyncBatchNorm, requires_grad=True),
            "in_channels": [384, 384, 384, 384],
            "in_index": [0, 1, 2, 3],
            "input_transform": "resize_concat",
            "channels": 1536,
            "pretrained_weights": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_ade20k_linear_head.pth",
        },
    }

    def __new__(cls, version: str, num_classes: int) -> NNFCNHead:
        """Constructor for FCNHead."""
        if version not in cls.FCNHEAD_CFG:
            msg = f"model type '{version}' is not supported"
            raise KeyError(msg)

        return NNFCNHead(**cls.FCNHEAD_CFG[version], num_classes=num_classes)
