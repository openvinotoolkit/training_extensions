"""OTX Anomaly Classification Dataset Item and Batch Class Definitions."""

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


@register_pytree_node  # Ideally, we should not use this decorator
@dataclass
class AnomalyClassificationDataItem(OTXDataEntity):
    """Anomaly classification dataset item."""

    @property
    def task(self) -> OTXTaskType:
        """Task type is anomaly classification."""
        return OTXTaskType.ANOMALY_CLASSIFICATION

    label: torch.LongTensor


@dataclass
class AnomalyClassificationDataBatch(OTXBatchDataEntity):
    """Anomaly classification batch."""

    labels: list[torch.LongTensor]

    # This is redundant. Task is already defined in AnomalyClassificationDatasetItem
    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.ANOMALY_CLASSIFICATION

    @classmethod
    def collate_fn(
        cls,
        entities: list[AnomalyClassificationDataItem],
        stack_images: bool = True,
    ) -> AnomalyClassificationDataBatch:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch = super().collate_fn(entities)
        images = tv_tensors.Image(data=torch.stack(tuple(batch.images), dim=0)) if stack_images else batch.images
        return AnomalyClassificationDataBatch(
            batch_size=batch.batch_size,
            images=images,
            imgs_info=batch.imgs_info,
            labels=[entity.label for entity in entities],
        )

    def pin_memory(self) -> AnomalyClassificationDataBatch:
        """Pin memory for member tensor variables."""
        return super().pin_memory().wrap(labels=[label.pin_memory() for label in self.labels])


@dataclass
class AnomalyClassificationPrediction(OTXPredEntity, AnomalyClassificationDataItem):
    """Anomaly classification Prediction item."""


@dataclass(kw_only=True)
class AnomalyClassificationBatchPrediction(OTXBatchPredEntity, AnomalyClassificationDataBatch):
    """Anomaly classification batch prediction."""

    anomaly_maps: torch.Tensor
