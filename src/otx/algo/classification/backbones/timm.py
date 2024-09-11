# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientNetV2 model.

Original papers:
- 'EfficientNetV2: Smaller Models and Faster Training,' https://arxiv.org/abs/2104.00298,
- 'Adversarial Examples Improve Image Recognition,' https://arxiv.org/abs/1911.09665.
"""
from __future__ import annotations

from typing import Literal

import timm
import torch
from torch import nn

TimmModelType = Literal[
    "mobilenetv3_large_100_miil_in21k",
    "mobilenetv3_large_100_miil",
    "tresnet_m",
    "tf_efficientnetv2_s.in21k",
    "tf_efficientnetv2_s.in21ft1k",
    "tf_efficientnetv2_m.in21k",
    "tf_efficientnetv2_m.in21ft1k",
    "tf_efficientnetv2_b0",
]


class TimmBackbone(nn.Module):
    """Timm backbone model."""

    def __init__(
        self,
        backbone: TimmModelType,
        pretrained: bool = False,
        pooling_type: str = "avg",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.pretrained: bool | dict = pretrained
        self.is_mobilenet = backbone.startswith("mobilenet")

        self.model = timm.create_model(
            self.backbone,
            pretrained=pretrained,
            num_classes=1000,
        )

        self.model.classifier = None  # Detach classifier. Only use 'backbone' part in otx.
        self.num_head_features = self.model.num_features
        self.num_features = self.model.conv_head.in_channels if self.is_mobilenet else self.model.num_features
        self.pooling_type = pooling_type

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        """Forward."""
        y = self.extract_features(x)
        return (y,)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        if self.is_mobilenet:
            x = self.model.conv_stem(x)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            return self.model.blocks(x)
        return self.model.forward_features(x)

    def get_config_optim(self, lrs: list[float] | float) -> list[dict[str, float]]:
        """Get optimizer configs."""
        parameters = [
            {"params": self.model.named_parameters()},
        ]
        if isinstance(lrs, list):
            for lr, param_dict in zip(lrs, parameters):
                param_dict["lr"] = lr
        else:
            for param_dict in parameters:
                param_dict["lr"] = lrs

        return parameters
