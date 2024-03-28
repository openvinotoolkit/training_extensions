# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX visual prompting data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchvision import tv_tensors

from otx.core.data.entity.base import (
    OTXBatchDataEntity,
    OTXBatchPredEntity,
    OTXDataEntity,
    OTXPredEntity,
    Points,
)
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
    labels: dict[str, LongTensor]
    polygons: list[Polygon]
    bboxes: tv_tensors.BoundingBoxes
    points: Points


@dataclass
class VisualPromptingPredEntity(OTXPredEntity, VisualPromptingDataEntity):
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
    labels: list[dict[str, LongTensor]]
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
        stack_images: bool = True,
    ) -> VisualPromptingBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader.

        Args:
            entities (list[VisualPromptingDataEntity]): List of VisualPromptingDataEntity objects.

        Returns:
            VisualPromptingBatchDataEntity: The collated batch data entity.
        """
        batch_data = super().collate_fn(entities, stack_images=stack_images)
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
        return (
            super()
            .pin_memory()
            .wrap(
                points=[
                    tv_tensors.wrap(point.pin_memory(), like=point) if point is not None else point
                    for point in self.points
                ],
                bboxes=[
                    tv_tensors.wrap(bbox.pin_memory(), like=bbox) if bbox is not None else bbox for bbox in self.bboxes
                ],
                masks=[tv_tensors.wrap(mask.pin_memory(), like=mask) for mask in self.masks],
                labels=[
                    {prompt_type: values.pin_memory() for prompt_type, values in labels.items()}
                    for labels in self.labels
                ],
            )
        )


@dataclass
class VisualPromptingBatchPredEntity(OTXBatchPredEntity, VisualPromptingBatchDataEntity):
    """Data entity to represent model output predictions for visual prompting task."""


@register_pytree_node
@dataclass
class ZeroShotVisualPromptingDataEntity(OTXDataEntity):
    """Data entity for zero-shot visual prompting task.

    Attributes:
        masks (tv_tensors.Mask): The masks of the instances.
        labels (LongTensor): The labels of the instances.
        polygons (list[Polygon]): The polygons of the instances.
        prompts (list[tv_tensors.TVTensor]): The prompts of the instances.
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING

    masks: tv_tensors.Mask
    labels: list[LongTensor]
    polygons: list[Polygon]
    prompts: list[tv_tensors.TVTensor]


@dataclass
class ZeroShotVisualPromptingBatchDataEntity(OTXBatchDataEntity[ZeroShotVisualPromptingDataEntity]):
    """Data entity for zero-shot visual prompting task.

    Attributes:
        masks (list[tv_tensors.Mask]): List of masks.
        labels (list[LongTensor]): List of labels.
        polygons (list[list[Polygon]]): List of polygons.
        prompts (list[list[tv_tensors.TVTensor]]): List of prompts.
    """

    masks: list[tv_tensors.Mask]
    labels: list[LongTensor]
    polygons: list[list[Polygon]]
    prompts: list[list[tv_tensors.TVTensor]]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING

    @classmethod
    def collate_fn(
        cls,
        entities: list[ZeroShotVisualPromptingDataEntity],
        stack_images: bool = True,
    ) -> ZeroShotVisualPromptingBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader.

        Args:
            entities (list[ZeroShotVisualPromptingDataEntity]): List of ZeroShotVisualPromptingDataEntity objects.

        Returns:
            ZeroShotVisualPromptingBatchDataEntity: The collated batch data entity.
        """
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        return ZeroShotVisualPromptingBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            masks=[entity.masks for entity in entities],
            labels=[entity.labels for entity in entities],
            polygons=[entity.polygons for entity in entities],
            prompts=[entity.prompts for entity in entities],
        )

    def pin_memory(self) -> ZeroShotVisualPromptingBatchDataEntity:
        """Pin memory for member tensor variables."""
        return (
            super()
            .pin_memory()
            .wrap(
                prompts=[
                    [tv_tensors.wrap(prompt.pin_memory(), like=prompt) for prompt in prompts]
                    for prompts in self.prompts
                ],
                masks=[tv_tensors.wrap(mask.pin_memory(), like=mask) for mask in self.masks],
                labels=[label.pin_memory() for label in self.labels],
            )
        )


@dataclass
class ZeroShotVisualPromptingBatchPredEntity(OTXBatchPredEntity, ZeroShotVisualPromptingBatchDataEntity):
    """Data entity to represent model output predictions for zero-shot visual prompting task."""

    prompts: list[Points]  # type: ignore[assignment]
