# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from mmaction.models.builder import HEADS
from mmaction.models.heads.base import BaseHead
from mmcv.cnn import normal_init

from otx.algorithms.action.adapters.mmaction.models.backbones.movinet import ConvBlock3D


@HEADS.register_module()
class MoViNetHead(BaseHead):
    """Classification head for MoViNet.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        hidden_dim (int): Number of channels in hidden layer.
        tf_like (bool): If True, uses TensorFlow-style padding. Default: False.
        conv_type (str): Type of convolutional layer. Default: '3d'.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Standard deviation for initialization. Default: 0.1.
        causal (bool): If True, uses causal convolution. Default: False.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_dim: int,
        tf_like: bool = False,
        conv_type: str = "3d",
        loss_cls=dict(type="CrossEntropyLoss"),
        spatial_type: str = "avg",
        dropout_ratio: float = 0.5,
        init_std: float = 0.1,
        causal: bool = False,
    ):
        super().__init__(num_classes, in_channels, loss_cls)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.classifier = nn.Sequential(
            ConvBlock3D(
                in_channels,
                hidden_dim,
                kernel_size=(1, 1, 1),
                tf_like=tf_like,
                causal=causal,
                conv_type=conv_type,
                bias=True,
            ),
            nn.SiLU(),
            nn.Dropout(p=0.2, inplace=True),
            ConvBlock3D(
                hidden_dim,
                num_classes,
                kernel_size=(1, 1, 1),
                tf_like=tf_like,
                causal=causal,
                conv_type=conv_type,
                bias=True,
            ),
        )

    def init_weights(self):
        """Initialize the parameters from scratch."""
        normal_init(self.classifier, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, T, H, W]
        cls_score = self.classifier(x)
        cls_score = cls_score.flatten(1)
        # [N, num_classes]
        return cls_score
