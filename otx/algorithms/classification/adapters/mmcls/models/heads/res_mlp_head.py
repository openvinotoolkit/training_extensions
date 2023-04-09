"""Module for defining Residual Multi-Layer-Perceptron head."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from torch import nn

from otx.algorithms.classification.adapters.mmcls.models.heads.custom_cls_head import (
    CustomLinearClsHead,
)


class ResMLPBlock(nn.Module):
    """ResMLPBlock class."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initializes the ResMLPBlock module.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.

        Returns:
            None.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResMLPBlock module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += identity
        return out


@HEADS.register_module()
class ResMLPHead(CustomLinearClsHead):
    """ResMLP Head class.

    Args:
        num_classes (int): The number of output classes.
        in_channels (int): The number of input channels.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        None.
    """

    def __init__(self, num_classes, in_channels, *args, **kwargs) -> None:
        """Initializes the ResMLPHead module.

        Args:
            num_classes (int): The number of output classes.
            in_channels (int): The number of input channels.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None.
        """
        super().__init__(num_classes, in_channels, *args, **kwargs)
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, 256)
        self.block1 = ResMLPBlock(256, 256)
        self.block2 = ResMLPBlock(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def fc(self, x):  # pylint: disable=method-hidden
        """Computes the fully connected layer of the ResMLPHead module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        breakpoint()
        x = x.view(-1, self.in_channels)
        x = F.relu(self.fc1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc2(x)
        return x
