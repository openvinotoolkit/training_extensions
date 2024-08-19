# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Custom activation implementation copied from mmcv.cnn.bricks.swish.py."""

from __future__ import annotations

from functools import partial
from typing import Callable

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


AVAILABLE_ACTIVATION_LIST: list[nn.Module] = [
    nn.ReLU,
    nn.LeakyReLU,
    nn.PReLU,
    nn.RReLU,
    nn.ReLU6,
    nn.ELU,
    nn.Sigmoid,
    nn.Tanh,
    nn.SiLU,
    nn.GELU,
    Swish,
]

ACTIVATION_LIST_NOT_SUPPORTING_INPLACE: list[nn.Module] = [
    nn.Tanh,
    nn.PReLU,
    nn.Sigmoid,
    Swish,
    nn.GELU,
    nn.SiLU,
]


def _get_act_type(activation: Callable[..., nn.Module]) -> type:
    """Get class type or name of given activation callable.

    Args:
        activation (Callable[..., nn.Module]): Activation layer module.

    Returns:
        (type): Class type of given activation callable.

    """
    return activation.func if isinstance(activation, partial) else activation  # type: ignore[return-value]


def build_activation_layer(
    activation: Callable[..., nn.Module] | nn.Module | None,
    inplace: bool = True,
) -> nn.Module | None:
    """Build activation layer.

    Args:
        activation (Callable[..., nn.Module]): Activation layer module.
            If None or pre-instanstiated module is given, return it as is.
            If callable is given, create the layer.
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.

    Returns:
        nn.Module: Created activation layer.
    """
    if activation is None or isinstance(activation, nn.Module):
        return activation

    if (layer_type := _get_act_type(activation)) not in AVAILABLE_ACTIVATION_LIST:
        msg = f"Unsupported activation: {layer_type.__name__}."
        raise ValueError(msg)

    layer = activation()

    # update inplace
    if layer.__class__ not in ACTIVATION_LIST_NOT_SUPPORTING_INPLACE:
        layer.inplace = inplace

    return layer
