# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Copy from mmpretrain/models/heads/vision_transformer_head.py."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional

from otx.algo.modules.base_module import BaseModule, Sequential
from otx.algo.utils.weight_init import trunc_normal_


class VisionTransformerClsHead(BaseModule):
    """Vision Transformer classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int, optional): Number of the dimensions for hidden layer.
            Defaults to None, which means no extra hidden layer.
        init_cfg (dict): The extra initialization configs. Defaults to
            ``dict(type='Constant', layer='Linear', val=0)``.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_dim: int | None = None,
        init_cfg: dict = {"type": "Constant", "layer": "Linear", "val": 0},  # noqa: B006
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        if self.num_classes <= 0:
            msg = f"num_classes={num_classes} must be a positive integer"
            raise ValueError(msg)

        self._init_layers()

    def _init_layers(self) -> None:
        """Init hidden layer if exists."""
        layers: list[tuple[str, nn.Module]]
        if self.hidden_dim is None:
            layers = [("head", nn.Linear(self.in_channels, self.num_classes))]
        else:
            layers = [
                ("pre_logits", nn.Linear(self.in_channels, self.hidden_dim)),
                ("act", nn.Tanh()),
                ("head", nn.Linear(self.hidden_dim, self.num_classes)),
            ]
        self.layers = Sequential(OrderedDict(layers))

    def init_weights(self) -> None:
        """Init weights of hidden layer if exists."""
        super().init_weights()
        # Modified from ClassyVision
        if hasattr(self.layers, "pre_logits"):
            # Lecun norm
            trunc_normal_(self.layers.pre_logits.weight, std=math.sqrt(1 / self.layers.pre_logits.in_features))
            nn.init.zeros_(self.layers.pre_logits.bias)

    def pre_logits(self, feats: tuple[list[torch.Tensor]]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of list of tensor, and each tensor is
        the feature of a backbone stage. In ``VisionTransformerClsHead``, we
        obtain the feature of the last stage and forward in hidden layer if
        exists.
        """
        feat = feats[-1]  # Obtain feature of the last scale.
        # For backward-compatibility with the previous ViT output
        cls_token = feat[-1] if isinstance(feat, list) else feat
        if self.hidden_dim is None:
            return cls_token
        x = self.layers.pre_logits(cls_token)
        return self.layers.act(x)

    def forward(self, feats: tuple[list[torch.Tensor]]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        return self.layers.head(pre_logits)

    def predict(
        self,
        feats: tuple[torch.Tensor],
    ) -> torch.Tensor:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.

        Returns:
            torch.Tensor: A tensor of softmax result.
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        return self._get_predictions(cls_score)

    def _get_predictions(self, cls_score: torch.Tensor) -> torch.Tensor:
        """Get the score from the classification score."""
        return functional.softmax(cls_score, dim=-1)
