# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINO-V2 model for the OTX classification."""

from __future__ import annotations
from otx.core.data.entity.base import OTXBatchLossEntity
import torch
from typing import TYPE_CHECKING, Any
from torch import nn
import torch.nn.functional as F
from typing import Any
from collections import OrderedDict
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
        
    def _create_model(self) -> nn.Module:
        """Create the model."""
        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=self.config.backbone
        )
        self.head = nn.Linear(
            self.config.head.in_channels,
            self.config.head.num_classes
        )
        return nn.Sequential(
            OrderedDict([
                ("backbone", self.backbone),
                ("head", self.head),
            ])
        )
         
    def forward(self, inputs: MulticlassClsBatchDataEntity):
        """Forward function.
        
        The output of the forward function should be the loss during training 
        and MulticlassBatchPredEntity during validation. 
        """
        customized_inputs = self._customize_inputs(inputs)
        
        feats = self.model(customized_inputs["x"])
        if self.training:
            outputs = self.loss(feats, customized_inputs["labels"])
        else:
            pred_scores = F.softmax(feats, dim=1)
            pred_labels = pred_scores.argmax(dim=1).cpu().detach().numpy()
            outputs = {
                "pred_scores": torch.max(pred_scores, dim=1).values.cpu().detach().numpy(),
                "pred_labels": pred_labels
            }
        return self._customize_outputs(outputs, inputs) 
    
    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        """Customize the inputs for the model."""
        inputs: dict[str, Any] = {}
        inputs["x"] = torch.stack([image for image in entity.images])
        inputs["labels"] = torch.cat([label for label in entity.labels])
        return inputs

    def _customize_outputs(
        self, 
        outputs: Any, 
        inputs: MulticlassClsBatchDataEntity
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
            labels=labels
        )