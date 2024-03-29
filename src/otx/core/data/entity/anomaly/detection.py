"""OTX Anomaly Detection Dataset Item and Batch Class Definitions."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision import tv_tensors

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
class AnomalyDetectionDataItem(OTXDataEntity):
    """Anomaly Detection dataset item."""

    @property
    def task(self) -> OTXTaskType:
        """Task type is anomaly detection."""
        return OTXTaskType.ANOMALY_DETECTION

    label: torch.LongTensor
    boxes: torch.Tensor
    mask: torch.Tensor


@dataclass
class AnomalyDetectionDataBatch(OTXBatchDataEntity):
    """Anomaly Detection batch."""

    labels: list[torch.LongTensor]
    boxes: torch.Tensor
    masks: torch.Tensor

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.ANOMALY_DETECTION

    @classmethod
    def collate_fn(
        cls,
        entities: list[AnomalyDetectionDataItem],
        stack_images: bool = True,
    ) -> AnomalyDetectionDataBatch:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch = super().collate_fn(entities)
        images = tv_tensors.Image(data=torch.stack(tuple(batch.images), dim=0)) if stack_images else batch.images
        return AnomalyDetectionDataBatch(
            batch_size=batch.batch_size,
            images=images,
            imgs_info=batch.imgs_info,
            labels=[entity.label for entity in entities],
            masks=torch.vstack([entity.mask for entity in entities]),
            boxes=[entity.boxes for entity in entities],
        )

    def pin_memory(self) -> AnomalyDetectionDataBatch:
        """Pin memory for member tensor variables."""
        return (
            super()
            .pin_memory()
            .wrap(
                labels=[label.pin_memory() for label in self.labels],
                masks=self.masks.pin_memory(),
                boxes=[box.pin_memory() for box in self.boxes],
            )
        )


@dataclass
class AnomalyDetectionPrediction(OTXPredEntity, AnomalyDetectionDataItem):
    """Anomaly Detection Prediction item."""


@dataclass(kw_only=True)
class AnomalyDetectionBatchPrediction(OTXBatchPredEntity, AnomalyDetectionDataBatch):
    """Anomaly classification batch prediction."""

    anomaly_maps: torch.Tensor
    box_scores: list[torch.Tensor]
    box_labels: list[torch.Tensor]
