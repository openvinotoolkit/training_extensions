# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX segmentation data entities."""

from __future__ import annotations

from dataclasses import dataclass

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
class SegDataEntity(OTXDataEntity):
    """Data entity for segmentation task.

    :param mask: mask annotations
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.SEMANTIC_SEGMENTATION

    masks: tv_tensors.Mask


@dataclass
class SegPredEntity(OTXPredEntity, SegDataEntity):
    """Data entity to represent the segmentation model output prediction."""


@dataclass
class SegBatchDataEntity(OTXBatchDataEntity[SegDataEntity]):
    """Data entity for segmentation task.

    :param masks: A list of  annotations
    """

    masks: list[tv_tensors.Mask]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.SEMANTIC_SEGMENTATION

    @classmethod
    def collate_fn(
        cls,
        entities: list[SegDataEntity],
        stack_images: bool = True,
    ) -> SegBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        return SegBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            masks=[entity.masks for entity in entities],
        )

    def pin_memory(self) -> SegBatchDataEntity:
        """Pin memory for member tensor variables."""
        return super().pin_memory().wrap(masks=[tv_tensors.wrap(mask.pin_memory(), like=mask) for mask in self.masks])


@dataclass
class SegBatchPredEntity(OTXBatchPredEntity, SegBatchDataEntity):
    """Data entity to represent model output predictions for segmentation task."""
