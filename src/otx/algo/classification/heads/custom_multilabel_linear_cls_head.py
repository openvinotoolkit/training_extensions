# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for defining multi-label linear classification head."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mmengine.model import normal_init
from mmpretrain.models.heads import MultiLabelLinearClsHead
from mmpretrain.registry import MODELS
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    from mmpretrain.structures import DataSample


@MODELS.register_module()
class CustomMultiLabelLinearClsHead(MultiLabelLinearClsHead):
    """Custom Linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        normalized (bool): Normalize input features and weights.
        scale (float): positive scale parameter.
        loss (dict): Config of classification loss.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        normalized: bool = False,
        scale: float = 1.0,
        loss: dict | None = None,
    ):
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
        super().__init__(
            loss=loss,
            num_classes=num_classes,
            in_channels=in_channels,
        )
        self.num_classes = num_classes
        self.normalized = normalized
        self.scale = scale
        self._init_layers()

    def _init_layers(self) -> None:
        if self.normalized:
            self.fc = AnglularLinear(self.in_channels, self.num_classes)
        else:
            self.fc = nn.Linear(self.in_channels, self.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights of head."""
        if isinstance(self.fc, nn.Linear):
            normal_init(self.fc, mean=0, std=0.01, bias=0)

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


class AnglularLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output cosine logits.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Init fuction of AngularLinear class."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight.data.normal_().renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward fuction of AngularLinear class."""
        cos_theta = functional.normalize(x.view(x.shape[0], -1), dim=1).mm(
            functional.normalize(self.weight.t(), p=2, dim=0),
        )
        return cos_theta.clamp(-1, 1)
