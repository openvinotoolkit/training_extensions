# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""This module contains the CustomMultiLabelNonLinearClsHead implementation for MMClassification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mmcv.cnn import build_activation_layer
from mmengine.model import constant_init, normal_init
from mmpretrain.models.heads import MultiLabelClsHead
from mmpretrain.registry import MODELS
from torch import nn

from .custom_multilabel_linear_cls_head import AnglularLinear

if TYPE_CHECKING:
    from mmpretrain.structures import DataSample


@MODELS.register_module()
class CustomMultiLabelNonLinearClsHead(MultiLabelClsHead):
    """Non-linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        hid_channels (int): Number of channels in the hidden feature map.
        act_cfg (dict | optional): The configuration of the activation function.
        scale (float): Positive scale parameter.
        loss (dict): Config of classification loss.
        dropout (bool): Whether use the dropout or not.
        normalized (bool): Normalize input features and weights in the last linar layer.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hid_channels: int = 1280,
        act_cfg: dict | None = None,
        scale: float = 1.0,
        loss: dict | None = None,
        dropout: bool = False,
        normalized: bool = False,
    ):
        act_cfg = act_cfg if act_cfg else {"type": "ReLU"}
        loss = (
            loss
            if loss
            else {
                "type": "CrossEntropyLoss",
                "use_sigmoid": True,
                "reduction": "mean",
                "loss_weight": 1.0,
            }
        )
        super().__init__(loss=loss)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hid_channels = hid_channels
        self.dropout = dropout
        self.normalized = normalized
        self.scale = scale

        if self.num_classes <= 0:
            msg = f"num_classes={num_classes} must be a positive integer"
            raise ValueError(msg)

        self._init_layers(act_cfg)

    def _init_layers(self, act_cfg: dict) -> None:
        """Initialize the layers."""
        modules = [
            nn.Linear(self.in_channels, self.hid_channels),
            nn.BatchNorm1d(self.hid_channels),
            build_activation_layer(act_cfg),
        ]
        if self.dropout:
            modules.append(nn.Dropout(p=0.2))
        if self.normalized:
            modules.append(AnglularLinear(self.hid_channels, self.num_classes))
        else:
            modules.append(nn.Linear(self.hid_channels, self.num_classes))

        self.classifier = nn.Sequential(*modules)
        self._init_weights()

    def _init_weights(self) -> None:
        """Iniitalize weights of model."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                normal_init(module, mean=0, std=0.01, bias=0)
            elif isinstance(module, nn.BatchNorm1d):
                constant_init(module, 1)

    def forward(self, feats: tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        return self.classifier(pre_logits)

    def loss(self, feats: tuple[torch.Tensor], data_samples: list[DataSample], **kwargs) -> dict:
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
        img_metas = [data_sample.metainfo for data_sample in data_samples]
        valid_label_mask = self.get_valid_label_mask(img_metas)
        cls_score = self(feats) * self.scale

        losses = super()._get_loss(cls_score, data_samples, valid_label_mask=valid_label_mask, **kwargs)
        losses["loss"] = losses["loss"] / self.scale
        return losses

    def get_valid_label_mask(self, img_metas: list[dict]) -> list[torch.Tensor]:
        """Get valid label mask using ignored_label."""
        valid_label_mask = []
        for meta in img_metas:
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
            valid_label_mask.append(mask)
        return torch.stack(valid_label_mask, dim=0)
