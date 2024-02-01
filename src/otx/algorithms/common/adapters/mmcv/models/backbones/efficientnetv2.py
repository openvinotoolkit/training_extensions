"""Implementation of EfficientNetV2.

Original papers:
- 'EfficientNetV2: Smaller Models and Faster Training,' https://arxiv.org/abs/2104.00298,
- 'Adversarial Examples Improve Image Recognition,' https://arxiv.org/abs/1911.09665.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=unused-argument, invalid-name

import os

import timm
from mmcv.runner import load_checkpoint
from torch import nn

from otx.utils.logger import get_logger

from ..builder import BACKBONES

logger = get_logger()

PRETRAINED_ROOT = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/"
pretrained_urls = {
    "efficientnetv2_s_21k": PRETRAINED_ROOT + "tf_efficientnetv2_s_21k-6337ad01.pth",
    "efficientnetv2_s_1k": PRETRAINED_ROOT + "tf_efficientnetv2_s_21ft1k-d7dafa41.pth",
}

NAME_DICT = {
    "mobilenetv3_large_21k": "mobilenetv3_large_100_miil_in21k",
    "mobilenetv3_large_1k": "mobilenetv3_large_100_miil",
    "tresnet": "tresnet_m",
    "efficientnetv2_s_21k": "tf_efficientnetv2_s_in21k",
    "efficientnetv2_s_1k": "tf_efficientnetv2_s_in21ft1k",
    "efficientnetv2_m_21k": "tf_efficientnetv2_m_in21k",
    "efficientnetv2_m_1k": "tf_efficientnetv2_m_in21ft1k",
    "efficientnetv2_b0": "tf_efficientnetv2_b0",
}


class TimmModelsWrapper(nn.Module):
    """Timm model wrapper."""

    def __init__(self, model_name, pretrained=False, pooling_type="avg", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained
        if model_name in ["mobilenetv3_large_100_miil_in21k", "mobilenetv3_large_100_miil"]:
            self.is_mobilenet = True
        else:
            self.is_mobilenet = False

        self.model = timm.create_model(NAME_DICT[self.model_name], pretrained=pretrained, num_classes=1000)
        if self.pretrained:
            logger.info(f"init weight - {pretrained_urls[self.model_name]}")
        self.model.classifier = None  # Detach classifier. Only use 'backbone' part in otx.
        self.num_head_features = self.model.num_features
        self.num_features = self.model.conv_head.in_channels if self.is_mobilenet else self.model.num_features
        self.pooling_type = pooling_type

    def forward(self, x, **kwargs):
        """Forward."""
        y = self.extract_features(x)
        return y

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


@BACKBONES.register_module()
class OTXEfficientNetV2(TimmModelsWrapper):
    """EfficientNetV2 for OTX."""

    def __init__(self, version="s_21k", **kwargs):
        self.model_name = "efficientnetv2_" + version
        super().__init__(model_name=self.model_name, **kwargs)

    def init_weights(self, pretrained=None):
        """Initialize weights."""
        if isinstance(pretrained, str) and os.path.exists(pretrained):
            load_checkpoint(self, pretrained)
            logger.info(f"init weight - {pretrained}")
        elif pretrained is not None:
            load_checkpoint(self, pretrained_urls[self.model_name])
            logger.info(f"init weight - {pretrained_urls[self.model_name]}")
