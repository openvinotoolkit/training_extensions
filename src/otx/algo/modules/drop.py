# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation replaces the functionality of mmcv.cnn.bricks.drop."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    Copy from mmcv.cnn.bricks.drop
    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DropPath module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying DropPath.
        """
        return drop_path(x, self.drop_prob, self.training)


class Dropout(nn.Dropout):
    """A wrapper for ``torch.nn.Dropout``.

    Copy from mmcv.cnn.bricks.drop
    We rename the ``p`` of ``torch.nn.Dropout`` to ``drop_prob``
    so as to be consistent with ``DropPath``

    Args:
        drop_prob (float): Probability of the elements to be
            zeroed. Default: 0.5.
        inplace (bool):  Do the operation inplace or not. Default: False.
    """

    def __init__(self, drop_prob: float = 0.5, inplace: bool = False):
        super().__init__(p=drop_prob, inplace=inplace)


DROPOUT_DICT = {
    "DropPath": DropPath,
    "Dropout": Dropout,
}


def build_dropout(cfg: dict, default_args: dict | None = None) -> Any:  # noqa: ANN401
    """Builder for drop out layers."""
    dropout_type = cfg.pop("type", None)
    if dropout_type is None:
        msg = "The cfg dict must contain the key 'type'"
        raise KeyError(msg)
    if dropout_type not in DROPOUT_DICT:
        msg = f"Cannot find {dropout_type} in {DROPOUT_DICT.keys()}"
        raise KeyError(msg)
    if default_args is not None:
        cfg.update(default_args)
    return DROPOUT_DICT[dropout_type](**cfg)
