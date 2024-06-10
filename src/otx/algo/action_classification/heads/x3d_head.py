# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""X3D head implementation."""
from __future__ import annotations

from torch import Tensor, nn

from otx.algo.action_classification.heads.base_head import BaseHead
from otx.algo.utils.weight_init import normal_init


class X3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (nn.module): Loss class like CrossEntropyLoss.
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        fc1_bias (bool): If the first fc layer has bias. Default: False.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_dim: int,
        loss_cls: nn.Module,
        spatial_type: str = "avg",
        dropout_ratio: float = 0.5,
        init_std: float = 0.01,
        fc1_bias: bool = False,
        average_clips: str | None = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            average_clips=average_clips,
        )  # Call the initializer of BaseHead

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc1_bias = fc1_bias

        self.fc1 = nn.Linear(self.in_channels, hidden_dim, bias=self.fc1_bias)
        self.fc2 = nn.Linear(hidden_dim, self.num_classes)

        self.relu = nn.ReLU()

        self.pool = None
        if self.spatial_type == "avg":
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif self.spatial_type == "max":
            self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        else:
            raise NotImplementedError

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc1, std=self.init_std)
        normal_init(self.fc2, std=self.init_std)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        # [N, in_channels, T, H, W]
        if self.pool is None:
            msg = "pool for X3DHead should be given."
            raise ValueError(msg)

        x = self.pool(x)
        # [N, in_channels, 1, 1, 1]
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        x = self.fc1(x)
        # [N, 2048]
        x = self.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        # [N, num_classes]
        return self.fc2(x)
