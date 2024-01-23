# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX visual prompting data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchvision import tv_tensors

from otx.core.data.entity.base import OTXBatchDataEntity, OTXBatchPredEntity, OTXDataEntity, OTXPredEntity, Points
from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from datumaro import Polygon
    from torch import LongTensor


@register_pytree_node
@dataclass
class VisualPromptingDataEntity(OTXDataEntity):
    """Data entity for visual prompting task.

    Attributes:
        masks (tv_tensors.Mask): The masks of the instances.
        labels (LongTensor): The labels of the instances.
        polygons (list[Polygon]): The polygons of the instances.
        bboxes (tv_tensors.BoundingBoxes): The bounding boxes of the instances.
        points (Points): The points of the instances.
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.VISUAL_PROMPTING

    masks: tv_tensors.Mask
    labels: list[LongTensor]
    polygons: list[Polygon]
    bboxes: tv_tensors.BoundingBoxes
    points: Points


@dataclass
class VisualPromptingPredEntity(VisualPromptingDataEntity, OTXPredEntity):
    """Data entity to represent the visual prompting model output prediction."""


@dataclass
class VisualPromptingBatchDataEntity(OTXBatchDataEntity[VisualPromptingDataEntity]):
    """Data entity for visual prompting task.

    Attributes:
        masks (list[tv_tensors.Mask]): List of masks.
        labels (list[LongTensor]): List of labels.
        polygons (list[list[Polygon]]): List of polygons.
        bboxes (list[tv_tensors.BoundingBoxes]): List of bounding boxes.
        points (list[Points]): List of points.
    """

    masks: list[tv_tensors.Mask]
    labels: list[LongTensor]
    polygons: list[list[Polygon]]
    bboxes: list[tv_tensors.BoundingBoxes]
    points: list[Points]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.VISUAL_PROMPTING

    @classmethod
    def collate_fn(
        cls,
        entities: list[VisualPromptingDataEntity],
    ) -> VisualPromptingBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader.

        Args:
            entities (list[VisualPromptingDataEntity]): List of VisualPromptingDataEntity objects.

        Returns:
            VisualPromptingBatchDataEntity: The collated batch data entity.
        """
        batch_data = super().collate_fn(entities)
        return VisualPromptingBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            masks=[entity.masks for entity in entities],
            labels=[entity.labels for entity in entities],
            polygons=[entity.polygons for entity in entities],
            points=[entity.points for entity in entities],
            bboxes=[entity.bboxes for entity in entities],
        )

    def pin_memory(self) -> VisualPromptingBatchDataEntity:
        """Pin memory for member tensor variables."""
        super().pin_memory()
        self.points = [tv_tensors.wrap(point.pin_memory(), like=point) if point is not None else point for point in self.points]
        self.bboxes = [tv_tensors.wrap(bbox.pin_memory(), like=bbox) if bbox is not None else bbox for bbox in self.bboxes]
        self.masks = [tv_tensors.wrap(mask.pin_memory(), like=mask) for mask in self.masks]
        self.labels = [label.pin_memory() for label in self.labels]
        return self


@dataclass
class VisualPromptingBatchPredEntity(VisualPromptingBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for visual prompting task."""
