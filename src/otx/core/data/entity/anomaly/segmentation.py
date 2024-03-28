"""OTX Anomaly Segmentation Dataset Item and Batch Class Definitions."""

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
        images = tv_tensors.Image(data=torch.stack(tuple(batch.images), dim=0)) if stack_images else batch.images
        return AnomalySegmentationDataBatch(
            batch_size=batch.batch_size,
            images=images,
            imgs_info=batch.imgs_info,
            labels=[entity.label for entity in entities],
            masks=torch.vstack([entity.mask for entity in entities]),
        )

    def pin_memory(self) -> AnomalySegmentationDataBatch:
        """Pin memory for member tensor variables."""
        return (
            super()
            .pin_memory()
            .wrap(
                labels=[label.pin_memory() for label in self.labels],
                masks=self.masks.pin_memory(),
            )
        )


@dataclass
class AnomalySegmentationPrediction(OTXPredEntity, AnomalySegmentationDataItem):
    """Anomaly Segmentation Prediction item."""


@dataclass(kw_only=True)
class AnomalySegmentationBatchPrediction(OTXBatchPredEntity, AnomalySegmentationDataBatch):
    """Anomaly classification batch prediction."""

    anomaly_maps: torch.Tensor
