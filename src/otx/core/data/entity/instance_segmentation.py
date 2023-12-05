# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX detection data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from otx.core.types.task import OTXTaskType

from .base import OTXBatchDataEntity, OTXBatchPredEntity, OTXDataEntity, OTXPredEntity

if TYPE_CHECKING:
    from datumaro import Polygon
    from torch import LongTensor
    from torchvision import tv_tensors


@dataclass
class InstanceSegDataEntity(OTXDataEntity):
    """Data entity for detection task.

    :param bboxes: Bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: Bbox labels as integer indices
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
class InstanceSegBatchDataEntity(OTXBatchDataEntity[InstanceSegDataEntity]):
    """Data entity for detection task.

    :param bboxes: A list of bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: A list of bbox labels as integer indices
    """

    bboxes: list[tv_tensors.BoundingBoxes]
    masks: list[tv_tensors.Mask]
    labels: list[LongTensor]
    polygons: list[Polygon]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.DETECTION

    @classmethod
    def collate_fn(cls, entities: list[InstanceSegDataEntity]) -> InstanceSegBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
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


@dataclass
class InstanceSegBatchPredEntity(InstanceSegBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for instance segmentation task."""
