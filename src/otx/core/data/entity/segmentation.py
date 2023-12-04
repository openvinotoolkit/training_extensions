# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX segmentation data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from otx.core.types.task import OTXTaskType

from .base import OTXBatchDataEntity, OTXBatchPredEntity, OTXDataEntity, OTXPredEntity

if TYPE_CHECKING:
    from torchvision import tv_tensors


@dataclass
class SegDataEntity(OTXDataEntity):
    """Data entity for segmentation task.

    :param gt_seg_map: mask annotations
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.SEMANTIC_SEGMENTATION

    gt_seg_map: tv_tensors.Mask

@dataclass
class SegPredEntity(SegDataEntity, OTXPredEntity):
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
    def collate_fn(cls, entities: list[SegDataEntity]) -> SegBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities)
        return SegBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            masks=[entity.gt_seg_map for entity in entities],
        )

@dataclass
class SegBatchPredEntity(SegBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for segmentation task."""
