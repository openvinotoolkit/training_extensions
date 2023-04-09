"""Module for defining CLIP Projection neck in mmcls."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
from mmcls.models.builder import NECKS
from mmcv.runner import BaseModule
from torch import nn


@NECKS.register_module()
class CLIPProjection(BaseModule):
    """CLIPProjection class.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        init_cfg (dict): The initialization configuration.

    Returns:
        None.
    """

    def __init__(self, in_channels, out_channels, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        scale = in_channels**-0.5
        self.proj = nn.Parameter(scale * torch.randn(in_channels, out_channels))

    def forward(self, inputs):
        """Performs a forward pass of the CLIPProjection module.

        Args:
            inputs (tuple or torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing the output tensor.
        """
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[-1]
            out = inputs @ self.proj
        elif isinstance(inputs, torch.Tensor):
            out = inputs @ self.proj
        else:
            raise TypeError("`CLIPProjection` neck inputs should be tuple or torch.tensor")
        return (out,)
