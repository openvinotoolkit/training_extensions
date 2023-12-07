# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINO-V2 model for the OTX classification."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity
from otx.core.model.entity.classification import OTXClassificationModel

if TYPE_CHECKING:
    from omegaconf import DictConfig

class DINOv2RegisterClassifier(OTXClassificationModel):
    """DINO-v2 Classification Model."""
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        super().__init__() # create the model

        self.loss = nn.CrossEntropyLoss()

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
        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=self.config.backbone.name,
        )
        if self.config.backbone.frozen:
            self._freeze_backbone(self.backbone)

        self.head = nn.Linear(
            self.config.head.in_channels,
            self.config.head.num_classes,
        )

        return nn.Sequential(
            OrderedDict([
                ("backbone", self.backbone),
                ("head", self.head),
            ]),
        )

    def _freeze_backbone(self, backbone: nn.Module) -> None:
        """Freeze the backbone."""
        for _, v in backbone.named_parameters():
            v.requires_grad = False

    def _preprocess_img(self, imgs: torch.Tensor) -> torch.Tensor:
        """Control normalize and BGR/RGB conversion."""
        # BGR -> RGB
        if self.data_preprocess_cfg.to_rgb and imgs.size(1) == 3:
            imgs = imgs.flip(1)

        # Normalization
        imgs = imgs.float()
        return (imgs - self.mean) / self.std

    def forward(
        self,
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        """Forward function.

        The output of the forward function should be the loss during training
        and MulticlassBatchPredEntity during validation.
        """
        customized_inputs = self._customize_inputs(inputs)
        feats = self.model(customized_inputs["x"])
        if self.training:
            outputs = self.loss(feats, customized_inputs["labels"])
        else:
            pred_scores = nn.functional.softmax(feats, dim=1)
            max_pred_elements, max_pred_idxs = torch.max(pred_scores, dim=1)
            pred_scores = max_pred_elements.cpu().detach().numpy()
            pred_labels = max_pred_idxs.cpu().detach().numpy()

            outputs = {
                "pred_scores": pred_scores,
                "pred_labels": pred_labels,
            }
        return self._customize_outputs(outputs, inputs)

    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        """Customize the inputs for the model."""
        inputs: dict[str, Any] = {}
        inputs["x"] = self._preprocess_img(
            torch.stack([torch.as_tensor(image) for image in entity.images]),
        )
        inputs["labels"] = torch.cat([torch.as_tensor(label) for label in entity.labels])
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

        batch_size = outputs["pred_labels"].shape[0]
        scores = []
        labels = []
        for b in range(batch_size):
            scores.append(outputs["pred_scores"][b])
            labels.append(outputs["pred_labels"][b])

        return MulticlassClsBatchPredEntity(
            batch_size=batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )
