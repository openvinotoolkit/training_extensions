# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Custom MoViNet Head for video recognition."""
from __future__ import annotations

from torch import Tensor, nn

from otx.algo.action_classification.backbones.movinet import ConvBlock3D
from otx.algo.action_classification.heads.base_head import BaseHead
from otx.algo.utils.weight_init import normal_init


class MoViNetHead(BaseHead):
    """Classification head for MoViNet.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        hidden_dim (int): Number of channels in hidden layer.
        tf_like (bool): If True, uses TensorFlow-style padding. Default: False.
        conv_type (str): Type of convolutional layer. Default: '3d'.
        loss_cls (nn.module): Loss class like CrossEntropyLoss.
        topk (tuple[int, int]): Top-K training loss calculation. Default: (1, 5).
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Standard deviation for initialization. Default: 0.1.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_dim: int,
        loss_cls: nn.Module,
        topk: tuple[int, int] = (1, 5),
        tf_like: bool = False,
        conv_type: str = "3d",
        average_clips: str | None = None,
    ):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            topk=topk,
            average_clips=average_clips,
        )  # Call the initializer of BaseHead

        self.init_std = 0.1
        self.classifier = nn.Sequential(
            ConvBlock3D(
                in_channels,
                hidden_dim,
                kernel_size=(1, 1, 1),
                tf_like=tf_like,
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
                conv_type=conv_type,
                bias=True,
            ),
        )

    def init_weights(self) -> None:
        """Initialize the parameters from scratch."""
        normal_init(self.classifier, std=self.init_std)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, T, H, W]
        cls_score = self.classifier(x)
        # [N, num_classes]
        return cls_score.flatten(1)
