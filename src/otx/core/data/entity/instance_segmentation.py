# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX instance segmentation data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchvision import tv_tensors

from otx.core.types.task import OTXTaskType

from .base import OTXBatchDataEntity, OTXBatchPredEntity, OTXBatchPredEntityWithXAI, OTXDataEntity, OTXPredEntity

if TYPE_CHECKING:
    from datumaro import Polygon
    from torch import LongTensor


@dataclass
class InstanceSegDataEntity(OTXDataEntity):
    """Data entity for instance segmentation task.

    Attributes:
        bboxes (tv_tensors.BoundingBoxes): The bounding boxes of the instances.
        masks (tv_tensors.Mask): The masks of the instances.
        labels (LongTensor): The labels of the instances.
        polygons (list[Polygon]): The polygons of the instances.
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.INSTANCE_SEGMENTATION

    bboxes: tv_tensors.BoundingBoxes
    masks: tv_tensors.Mask
    labels: LongTensor
    polygons: list[Polygon]


@dataclass
class InstanceSegPredEntity(InstanceSegDataEntity, OTXPredEntity):
    """Data entity to represent the detection model output prediction."""


@dataclass
class InstanceSegPredEntityWithXAI(InstanceSegDataEntity, OTXBatchPredEntityWithXAI):
    """Data entity to represent the detection model output prediction with explanation."""


@dataclass
class InstanceSegBatchDataEntity(OTXBatchDataEntity[InstanceSegDataEntity]):
    """Batch entity for InstanceSegDataEntity.

    Attributes:
        bboxes (list[tv_tensors.BoundingBoxes]): List of bounding boxes.
        masks (list[tv_tensors.Mask]): List of masks.
        labels (list[LongTensor]): List of labels.
        polygons (list[list[Polygon]]): List of polygons.
    """

    bboxes: list[tv_tensors.BoundingBoxes]
    masks: list[tv_tensors.Mask]
    labels: list[LongTensor]
    polygons: list[list[Polygon]]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.INSTANCE_SEGMENTATION

    @classmethod
    def collate_fn(cls, entities: list[InstanceSegDataEntity]) -> InstanceSegBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader.

        Args:
            entities (list[InstanceSegDataEntity]): List of InstanceSegDataEntity objects.

        Returns:
            InstanceSegBatchDataEntity: The collated batch data entity.
        """
        batch_data = super().collate_fn(entities)
        return InstanceSegBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            bboxes=[entity.bboxes for entity in entities],
            masks=[entity.masks for entity in entities],
            labels=[entity.labels for entity in entities],
            polygons=[entity.polygons for entity in entities],
        )

    def pin_memory(self) -> InstanceSegBatchDataEntity:
        """Pin memory for member tensor variables."""
        super().pin_memory()
        self.bboxes = [tv_tensors.wrap(bbox.pin_memory(), like=bbox) for bbox in self.bboxes]
        self.masks = [tv_tensors.wrap(mask.pin_memory(), like=mask) for mask in self.masks]
        self.labels = [label.pin_memory() for label in self.labels]
        return self


@dataclass
class InstanceSegBatchPredEntity(InstanceSegBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for instance segmentation task."""


@dataclass
class InstanceSegBatchPredEntityWithXAI(InstanceSegBatchDataEntity, OTXBatchPredEntityWithXAI):
    """Data entity to represent model output predictions for instance segmentation task with explanations."""
