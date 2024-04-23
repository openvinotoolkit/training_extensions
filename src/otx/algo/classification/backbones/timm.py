# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientNetV2 model.

Original papers:
- 'EfficientNetV2: Smaller Models and Faster Training,' https://arxiv.org/abs/2104.00298,
- 'Adversarial Examples Improve Image Recognition,' https://arxiv.org/abs/1911.09665.
"""


import os

import torch
import timm
from otx.algo.utils.mmengine_utils import load_from_http, load_checkpoint_to_model
from torch import nn
from typing import Literal

PRETRAINED_ROOT = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/"
pretrained_urls = {
    "efficientnetv2_s_21k": PRETRAINED_ROOT + "tf_efficientnetv2_s_21k-6337ad01.pth",
    "efficientnetv2_s_1k": PRETRAINED_ROOT + "tf_efficientnetv2_s_21ft1k-d7dafa41.pth",
}

TIMM_MODEL_NAME_DICT = {
    "mobilenetv3_large_21k": "mobilenetv3_large_100_miil_in21k",
    "mobilenetv3_large_1k": "mobilenetv3_large_100_miil",
    "tresnet": "tresnet_m",
    "efficientnetv2_s_21k": "tf_efficientnetv2_s_in21k",
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
    def __init__(
        self,
        backbone: TimmModelType,
        pretrained=False,
        pooling_type="avg",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.pretrained = pretrained
        self.is_mobilenet = backbone.startswith("mobilenet")

        self.model = timm.create_model(TIMM_MODEL_NAME_DICT[self.backbone], pretrained=pretrained, num_classes=1000)
        if self.pretrained:
            print(f"init weight - {pretrained_urls[self.backbone]}")
        self.model.classifier = None  # Detach classifier. Only use 'backbone' part in otx.
        self.num_head_features = self.model.num_features
        self.num_features = self.model.conv_head.in_channels if self.is_mobilenet else self.model.num_features
        self.pooling_type = pooling_type

    def forward(self, x, **kwargs):
        """Forward."""
        y = self.extract_features(x)
        return (y,)

    def extract_features(self, x):
        """Extract features."""
        if self.is_mobilenet:
            x = self.model.conv_stem(x)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            y = self.model.blocks(x)
            return y
        return self.model.forward_features(x)

    def get_config_optim(self, lrs):
        """Get optimizer configs."""
        parameters = [
            {"params": self.model.named_parameters()},
        ]
        if isinstance(lrs, list):
            assert len(lrs) == len(parameters)
            for lr, param_dict in zip(lrs, parameters):
                param_dict["lr"] = lr
        else:
            assert isinstance(lrs, float)
            for param_dict in parameters:
                param_dict["lr"] = lrs

        return parameters

    def init_weights(self, pretrained: str | bool | None = None):
        """Initialize weights."""
        checkpoint = None
        if isinstance(pretrained, str) and os.path.exists(pretrained):
            checkpoint = torch.load(pretrained, None)
            print(f"init weight - {pretrained}")
        elif pretrained is not None:
            checkpoint = load_from_http(pretrained_urls[self.key])
            print(f"init weight - {pretrained_urls[self.key]}")
        if checkpoint is not None:
            load_checkpoint_to_model(self, checkpoint)
