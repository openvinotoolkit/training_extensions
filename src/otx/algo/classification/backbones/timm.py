# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Timm Backbone Class for OTX classification.

Original papers:
- 'EfficientNetV2: Smaller Models and Faster Training,' https://arxiv.org/abs/2104.00298,
- 'Adversarial Examples Improve Image Recognition,' https://arxiv.org/abs/1911.09665.
"""
from __future__ import annotations

import timm
import torch
from torch import nn


class TimmBackbone(nn.Module):
    """Timm backbone model.

    Args:
        model_name (str): The name of the model.
            You can find available models at timm.list_models() or timm.list_pretrained().
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to False.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.pretrained: bool | dict = pretrained

        self.model = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=1000,
        )

        self.model.classifier = None  # Detach classifier. Only use 'backbone' part in otx.
        self.num_head_features = self.model.num_features
        self.num_features = self.model.num_features

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        """Forward."""
        y = self.extract_features(x)
        return (y,)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
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
