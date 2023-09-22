"""Contrastive head to get contrastive loss with predictor head."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# pylint: disable=unused-argument
from typing import Any, Dict

import torch
from mmpretrain.models.builder import HEADS, build_neck
from torch import nn
from torch.nn import functional


@HEADS.register_module()
class ConstrastiveHead(nn.Module):
    """Head for contrastive learning.

    Args:
        predictor (dict): configurations for predictor.
        size_average (bool): whether averaging loss using batch size. Default value is True.
    """

    def __init__(self, predictor: Dict[str, Any], size_average: bool = True, **kwargs) -> None:
        super().__init__()
        self.predictor = build_neck(predictor)
        self.size_average = size_average

    def init_weights(self, init_linear: str = "normal") -> None:
        """Initialize predictor weights.

        Args:
            init_linear (str): Option to initialize weights. Default: 'normal'
        """
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """Forward head.

        Args:
            inputs (Tensor): NxC input features.
            targets (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor(inputs)
        pred_norm = functional.normalize(pred, dim=1)
        target_norm = functional.normalize(targets, dim=1)
        loss = 2 * inputs.size(0) - 2 * (pred_norm * target_norm).sum()
        if self.size_average:
            loss /= inputs.size(0)

        return {"loss": loss}
