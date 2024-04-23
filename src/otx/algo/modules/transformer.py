"""MMCV Transformer modules."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import copy

import torch
import torch.nn.functional
from mmengine.model import BaseModule, Sequential
from timm.models.layers import DropPath
from torch import nn

from otx.algo.modules.activation import build_activation_layer


class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        layer_scale_init_value (float): Initial value of scale factor in
            LayerScale. Default: 1.0
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        act_cfg: dict | None = None,
        ffn_drop: float = 0.0,
        dropout_layer: dict | None = None,
        add_identity: bool = True,
        init_cfg: dict | None = None,
    ):
        super().__init__(init_cfg)
        if num_fcs < 2:
            msg = "The number of fully-connected layers in FFNs should be at least 2."
            raise ValueError(msg)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        if act_cfg is None:
            act_cfg = {"type": "ReLU", "inplace": True}

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    build_activation_layer(act_cfg),
                    nn.Dropout(ffn_drop),
                ),
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)

        if dropout_layer:
            _dropout_layer = copy.deepcopy(dropout_layer)
            dropout_type = _dropout_layer.pop("type")
            if dropout_type != "DropPath":
                msg = f"Unsupported dropout type {dropout_type}"
                raise NotImplementedError(msg)
            self.dropout_layer = DropPath(**_dropout_layer)
        else:
            self.dropout_layer = torch.nn.Identity()

        self.add_identity = add_identity
        self.gamma2 = nn.Identity()

    def forward(self, x: torch.Tensor, identity: torch.Tensor | None = None) -> torch.Tensor:
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        out = self.gamma2(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
