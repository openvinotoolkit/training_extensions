"""Multi-layer Perceptron (MLP) for Self-supervised learning methods.

This MLP consists of fc (conv) - norm - relu - fc (conv).
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=dangerous-default-value
from typing import Any, Dict

import torch
from mmcv.cnn import build_norm_layer, kaiming_init, normal_init
from mmseg.models.builder import NECKS  # pylint: disable=no-name-in-module
from torch import nn


@NECKS.register_module()
class SelfSLMLP(nn.Module):
    """The SelfSLMLP neck: fc/conv-bn-relu-fc/conv.

    Args:
        in_channels (int): The number of feature output channels from backbone.
        hid_channels (int): The number of channels for a hidden layer.
        out_channels (int): The number of output channels of SelfSLMLP.
        norm_cfg (dict): Normalize configuration. Default: dict(type="BN1d").
        use_conv (bool): Whether using conv instead of fc. Default: False.
        with_avg_pool (bool): Whether using average pooling before passing MLP.
                              Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        norm_cfg: Dict[str, Any] = dict(type="BN1d"),
        use_conv: bool = False,
        with_avg_pool: bool = True,
    ):
        super().__init__()

        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.use_conv = use_conv
        if use_conv:
            self.mlp = nn.Sequential(
                nn.Conv2d(in_channels, hid_channels, 1),
                build_norm_layer(norm_cfg, hid_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(hid_channels, out_channels, 1),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hid_channels),
                build_norm_layer(norm_cfg, hid_channels)[1],
                nn.ReLU(inplace=True),
                nn.Linear(hid_channels, out_channels),
            )

    def init_weights(self, init_linear: str = "normal", std: float = 0.01, bias: float = 0.0):
        """Initialize SelfSLMLP weights.

        Args:
            init_linear (str): Option to initialize weights. Default: "normal".
            std (float): Standard deviation for normal initialization. Default: 0.01.
            bias (float): Bias for normal initialization. Default: 0.
        """
        if init_linear not in ["normal", "kaiming"]:
            raise ValueError(f"Undefined init_linear: {init_linear}")
        for m in self.modules():  # pylint: disable=invalid-name
            if isinstance(m, nn.Linear):
                if init_linear == "normal":
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward SelfSLMLP.

        Args:
            x (Tensor, tuple, list): Inputs to pass MLP.
                If a type of the inputs is tuple or list, just use the last index.

        Return:
            Tensor: Features passed SelfSLMLP.
        """
        if isinstance(x, (tuple, list)):
            # using last output
            x = x[-1]
        if not isinstance(x, torch.Tensor):
            raise TypeError("neck inputs should be tuple or torch.tensor")
        if self.with_avg_pool:
            x = self.avgpool(x)
        if self.use_conv:  # pylint: disable=no-else-return
            return self.mlp(x)
        else:
            return self.mlp(x.view(x.size(0), -1))
