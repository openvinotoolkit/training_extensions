# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Linear Head Implementation.

The original source code is mmpretrain.models.heads.linear_head.LinearClsHead.
you can refer https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/heads/linear_head.py

"""


from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional

from otx.algo.modules.base_module import BaseModule


class LinearClsHead(BaseModule):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        init_cfg: dict = {"type": "Normal", "layer": "Linear", "std": 0.01},  # noqa: B006
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            msg = f"num_classes={num_classes} must be a positive integer"
            raise ValueError(msg)

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, feats: tuple[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """The forward process."""
        if isinstance(feats, tuple):
            feats = feats[-1]
        # The final classification head.
        return self.fc(feats)

    def predict(
        self,
        feats: tuple[torch.Tensor],
        **kwargs,
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
        return functional.softmax(cls_score, dim=1)
