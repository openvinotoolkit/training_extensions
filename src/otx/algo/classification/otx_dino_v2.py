# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINO-V2 model for the OTX classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity
from otx.core.model.entity.classification import OTXClassificationModel

if TYPE_CHECKING:
    from omegaconf import DictConfig

class DINOv2(nn.Module):
    """DINO-v2 Model."""
    def __init__(
        self,
        backbone_name: str,
        freeze_backbone: bool,
        head_in_channels: int,
        num_classes: int,
        training: bool,
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

        self.training = training

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

class DINOv2RegisterClassifier(OTXClassificationModel):
    """DINO-v2 Classification Model with register."""
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        super().__init__() # create the model

        # NOTE,
        # We've decided to use MMpretrain's pipeline for this model
        # It's hard to use ClsDataPreprocessor since the model is not related to MMpretrain
        # That's the reason why I implemented the below preprocess things
        self.data_preprocess_cfg = self.config.data_preprocess
        self.register_buffer(
            'mean', torch.tensor(self.data_preprocess_cfg.mean).view(-1, 1, 1), False,
        )
        self.register_buffer(
            'std', torch.tensor(self.data_preprocess_cfg.std).view(-1, 1, 1), False,
        )

    def _create_model(self) -> nn.Module:
        """Create the model."""
        return DINOv2(
            backbone_name=self.config.backbone.name,
            freeze_backbone=self.config.backbone.frozen,
            head_in_channels=self.config.head.in_channels,
            num_classes=self.config.head.num_classes,
            training=self.training,
        )

    def _preprocess_img(self, imgs: torch.Tensor) -> torch.Tensor:
        """Control normalize and BGR/RGB conversion."""
        # BGR -> RGB
        if self.data_preprocess_cfg.to_rgb and imgs.size(1) == 3:
            imgs = imgs.flip(1)
        return (imgs - self.mean) / self.std


    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        """Customize the inputs for the model."""
        inputs: dict[str, Any] = {}
        inputs["imgs"] = self._preprocess_img(torch.stack(entity.images))
        inputs["labels"] = torch.cat(entity.labels)
        return inputs

    def _customize_outputs(
        self,
        outputs: Any, # noqa: ANN401
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
        pred_scores = max_pred_elements.detach()
        pred_labels = max_pred_idxs.detach()

        scores = torch.unbind(pred_scores, dim=0)
        labels = torch.unbind(pred_labels, dim=0)

        return MulticlassClsBatchPredEntity(
            batch_size=pred_labels.shape[0],
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )
