# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation of ChannelAttention copied from mmdet.models.layers.se_layer.py."""

import copy

import torch
from mmdet.utils import OptMultiConfig  # TODO (sungchul): remove types
from torch import Tensor, nn


class ChannelAttention(nn.Module):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self, channels: int, init_cfg: OptMultiConfig = None) -> None:
        super().__init__()
        # from mmengine.model.BaseModule
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out
