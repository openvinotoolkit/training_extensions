# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation replaces the functionality of mmcv.cbricks.activation.build_activation_layer."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Swish(nn.Module):
    """Swish Module.

    This module applies the swish function:

    .. math::
        Swish(x) = x * Sigmoid(x)

    Returns:
        Tensor: The output tensor.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        return x * torch.sigmoid(x)


AVAILABLE_ACTIVATION_LIST: list[str] = [
    "ReLU",
    "LeakyReLU",
    "PReLU",
    "RReLU",
    "ReLU6",
    "ELU",
    "Sigmoid",
    "Tanh",
    "SiLU",
    "GELU",
    "Swish",
]
