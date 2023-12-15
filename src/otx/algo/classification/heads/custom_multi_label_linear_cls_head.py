"""Module for defining multi-label linear classification head."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mmpretrain.registry import MODELS
from mmpretrain.models.heads import MultiLabelLinearClsHead
from mmengine.model import normal_init
from torch import nn

if TYPE_CHECKING:
    from mmpretrain.structures import DataSample 

@MODELS.register_module()
class CustomMultiLabelLinearClsHead(MultiLabelLinearClsHead):
    """Custom Linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        normalized (bool): Normalize input features and weights.
        scale (float): positive scale parameter.
        loss (dict): Config of classification loss.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        normalized=False,
        scale=1.0,
        loss=None,
    ):
        loss = loss if loss else dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=1.0)
        super().__init__(
            loss=loss,
            num_classes=num_classes,
            in_channels=in_channels
        )
        self.num_classes = num_classes
        self.normalized = normalized
        self.scale = scale
        self._init_layers()

    def _init_layers(self):
        if self.normalized:
            self.fc = AnglularLinear(self.in_channels, self.num_classes)
        else:
            self.fc = nn.Linear(self.in_channels, self.num_classes)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights of head."""
        if isinstance(self.fc, nn.Linear):
            normal_init(self.fc, mean=0, std=0.01, bias=0)

    def loss(self, feats: tuple[torch.Tensor], data_samples: list[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = super().loss(feats, data_samples, **kwargs)
        losses["loss"] =  losses["loss"]/ self.scale
        print(losses) 
        return losses
    
class AnglularLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output cosine logits.
    """

    def __init__(self, in_features, out_features):
        """Init fuction of AngularLinear class."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight.data.normal_().renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(self, x):
        """Forward fuction of AngularLinear class."""
        cos_theta = F.normalize(x.view(x.shape[0], -1), dim=1).mm(F.normalize(self.weight.t(), p=2, dim=0))
        return cos_theta.clamp(-1, 1)
