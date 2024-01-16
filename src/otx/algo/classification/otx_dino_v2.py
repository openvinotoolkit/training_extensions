# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINO-V2 model for the OTX classification."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
)
from otx.core.model.entity.classification import OTXMulticlassClsModel


class DINOv2(nn.Module):
    """DINO-v2 Model."""

    def __init__(
        self,
        backbone_name: str,
        freeze_backbone: bool,
        head_in_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=backbone_name,
        )

        if freeze_backbone:
            self._freeze_backbone(self.backbone)

        self.head = nn.Linear(
            head_in_channels,
            num_classes,
        )

        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()

    def _freeze_backbone(self, backbone: nn.Module) -> None:
        """Freeze the backbone."""
        for _, v in backbone.named_parameters():
            v.requires_grad = False

    def forward(self, imgs: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Forward function."""
        feats = self.backbone(imgs)
        logits = self.head(feats)
        if self.training:
            return self.loss(logits, labels)
        return self.softmax(logits)


class DINOv2RegisterClassifier(OTXMulticlassClsModel):
    """DINO-v2 Classification Model with register."""

    def __init__(self, num_classes: int, config: DictConfig | dict) -> None:
        self.config = DictConfig(config)
        super().__init__(num_classes=num_classes)  # create the model

    def _create_model(self) -> nn.Module:
        """Create the model."""
        return DINOv2(
            backbone_name=self.config.backbone.name,
            freeze_backbone=self.config.backbone.frozen,
            head_in_channels=self.config.head.in_channels,
            num_classes=self.config.head.num_classes,
        )

    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        """Customize the inputs for the model."""
        return {
            "imgs": entity.stacked_images,
            "labels": torch.cat(entity.labels),
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        """Customize the outputs for the model."""
        if self.training:
            if not isinstance(outputs, torch.Tensor):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            losses["loss"] = outputs
            return losses

        max_pred_elements, max_pred_idxs = torch.max(outputs, dim=1)
        pred_scores = max_pred_elements
        pred_labels = max_pred_idxs

        scores = torch.unbind(pred_scores, dim=0)
        labels = torch.unbind(pred_labels, dim=0)

        return MulticlassClsBatchPredEntity(
            batch_size=pred_labels.shape[0],
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )
