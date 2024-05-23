# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX instance segmentation data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchvision import tv_tensors

from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType

from .base import OTXBatchDataEntity, OTXBatchPredEntity, OTXDataEntity, OTXPredEntity

if TYPE_CHECKING:
    from datumaro import Polygon
    from torch import LongTensor


@register_pytree_node
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
class InstanceSegPredEntity(OTXPredEntity, InstanceSegDataEntity):
    """Data entity to represent the detection model output prediction."""


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
    def collate_fn(
        cls,
        entities: list[InstanceSegDataEntity],
        stack_images: bool = True,
    ) -> InstanceSegBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader.

        Args:
            entities (list[InstanceSegDataEntity]): List of InstanceSegDataEntity objects.
            stack_images: If True, return 4D B x C x H x W image tensor.
                Otherwise return a list of 3D C x H x W image tensor.

        Returns:
            InstanceSegBatchDataEntity: The collated batch data entity.
        """
        batch_data = super().collate_fn(entities, stack_images=stack_images)
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
        return (
            super()
            .pin_memory()
            .wrap(
                bboxes=[tv_tensors.wrap(bbox.pin_memory(), like=bbox) for bbox in self.bboxes],
                masks=[tv_tensors.wrap(mask.pin_memory(), like=mask) for mask in self.masks],
                labels=[label.pin_memory() for label in self.labels],
            )
        )


@dataclass
class InstanceSegBatchPredEntity(OTXBatchPredEntity, InstanceSegBatchDataEntity):
    """Data entity to represent model output predictions for instance segmentation task."""
