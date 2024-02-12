"""OTX Anomaly Segmentation Dataset Item and Batch Class Definitions."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision import tv_tensors
from torchvision.transforms.functional import resize

from otx.core.data.entity.base import (
    OTXBatchDataEntity,
    OTXBatchPredEntity,
    OTXDataEntity,
    OTXPredEntity,
)
from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType


@register_pytree_node
@dataclass
class AnomalySegmentationDataItem(OTXDataEntity):
    """Anomaly segmentation dataset item."""

    @property
    def task(self) -> OTXTaskType:
        """Task type is anomaly segmentation."""
        return OTXTaskType.ANOMALY_SEGMENTATION

    label: torch.LongTensor
    mask: torch.Tensor


@dataclass
class AnomalySegmentationDataBatch(OTXBatchDataEntity):
    """Anomaly Segmentation batch."""

    labels: list[torch.LongTensor]
    masks: torch.Tensor

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.ANOMALY_SEGMENTATION

    @classmethod
    def collate_fn(
        cls,
        entities: list[AnomalySegmentationDataItem],
        stack_images: bool = True,
    ) -> AnomalySegmentationDataBatch:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch = super().collate_fn(entities)
        images = tv_tensors.Image(data=torch.stack(batch.images, dim=0)) if stack_images else batch.images
        return AnomalySegmentationDataBatch(
            batch_size=batch.batch_size,
            images=images,
            imgs_info=batch.imgs_info,
            labels=[entity.label for entity in entities],
            masks=torch.vstack([resize(entity.mask, size=[images.shape[2], images.shape[3]]) for entity in entities]),
        )

    def pin_memory(self) -> AnomalySegmentationDataBatch:
        """Pin memory for member tensor variables."""
        super().pin_memory()
        self.labels = [label.pin_memory() for label in self.labels]
        self.masks = self.masks.pin_memory()
        return self


@dataclass
class AnomalySegmentationPrediction(AnomalySegmentationDataItem, OTXPredEntity):
    """Anomaly Segmentation Prediction item."""


@dataclass
class AnomalySegmentationBatchPrediction(AnomalySegmentationDataBatch, OTXBatchPredEntity):
    """Anomaly classification batch prediction."""

    anomaly_maps: torch.Tensor
    # Note: ideally this should be anomalous_scores but it is now used to shadow the scores in OTXBatchPredEntity
    scores: torch.bool
