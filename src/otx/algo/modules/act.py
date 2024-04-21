# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation replaces the functionality of mmcv.cnn.bricks.activation.build_activation_layer."""

import torch
from torch import nn


class Swish(nn.Module):
    """Swish Module.
    This module applies the swish function:
    .. math::
        Swish(x) = x * Sigmoid(x)
    Returns:
        Tensor: The output tensor.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        """
        return x * torch.sigmoid(x)


ACTIVATION_DICT = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "PReLU": nn.PReLU,
    "RReLU": nn.RReLU,
    "ReLU6": nn.ReLU6,
    "ELU": nn.ELU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "SiLU": nn.SiLU,
    "GELU": nn.GELU,
    "Swish": Swish,
}


def build_activation_layer(cfg: dict) -> nn.Module:
    """Build activation layer.
    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
    Returns:
        nn.Module: Created activation layer.
    """
    cfg_copy = cfg.copy()
    activation_type = cfg_copy.pop("type", None)
    if activation_type is None:
        msg = "The cfg dict must contain the key 'type'"
        raise KeyError(msg)
    if activation_type not in ACTIVATION_DICT:
        msg = f"Cannot find {activation_type} in {ACTIVATION_DICT.keys()}"
        raise KeyError(msg)

    return ACTIVATION_DICT[activation_type](**cfg_copy)
