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

from otx.algo.utils.mmengine_utils import load_from_http

PRETRAINED_ROOT = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/"
pretrained_urls = {
    "efficientnetv2_s_21k": PRETRAINED_ROOT + "tf_efficientnetv2_s_21k-6337ad01.pth",
    "efficientnetv2_s_1k": PRETRAINED_ROOT + "tf_efficientnetv2_s_21ft1k-d7dafa41.pth",
}

TIMM_MODEL_NAME_DICT = {
    "mobilenetv3_large_21k": "mobilenetv3_large_100_miil_in21k",
    "mobilenetv3_large_1k": "mobilenetv3_large_100_miil",
    "tresnet": "tresnet_m",
    "efficientnetv2_s_21k": "tf_efficientnetv2_s.in21k",
    "efficientnetv2_s_1k": "tf_efficientnetv2_s_in21ft1k",
    "efficientnetv2_m_21k": "tf_efficientnetv2_m_in21k",
    "efficientnetv2_m_1k": "tf_efficientnetv2_m_in21ft1k",
    "efficientnetv2_b0": "tf_efficientnetv2_b0",
}

TimmModelType = Literal[
    "mobilenetv3_large_21k",
    "mobilenetv3_large_1k",
    "tresnet",
    "efficientnetv2_s_21k",
    "efficientnetv2_s_1k",
    "efficientnetv2_m_21k",
    "efficientnetv2_m_1k",
    "efficientnetv2_b0",
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
        if pretrained and self.backbone in pretrained_urls:
            # This pretrained weight is saved into ~/.cache/torch/hub/checkpoints
            # Otherwise, it is stored in ~/.cache/huggingface/hub. (timm defaults)
            self.pretrained = load_from_http(filename=pretrained_urls[self.backbone])

        self.model = timm.create_model(
            TIMM_MODEL_NAME_DICT[self.backbone],
            pretrained=self.pretrained,
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
