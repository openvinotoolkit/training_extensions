# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX detection data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchvision import tv_tensors

from otx.core.data.entity.base import (
    OTXBatchDataEntity,
    OTXBatchPredEntity,
    OTXBatchPredEntityWithXAI,
    OTXDataEntity,
    OTXPredEntity,
    OTXPredEntityWithXAI,
)
from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from torch import LongTensor


@register_pytree_node
@dataclass
class DetDataEntity(OTXDataEntity):
    """Data entity for detection task.

    :param bboxes: Bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: Bbox labels as integer indices
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.DETECTION

    bboxes: tv_tensors.BoundingBoxes
    labels: LongTensor


@dataclass
class DetPredEntity(DetDataEntity, OTXPredEntity):
    """Data entity to represent the detection model output prediction."""


@dataclass
class DetPredEntityWithXAI(DetDataEntity, OTXPredEntityWithXAI):
    """Data entity to represent the detection model output prediction with explanations."""


@dataclass
class DetBatchDataEntity(OTXBatchDataEntity[DetDataEntity]):
    """Data entity for detection task.

    :param bboxes: A list of bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: A list of bbox labels as integer indices
    """

    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[LongTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.DETECTION

    @classmethod
    def collate_fn(cls, entities: list[DetDataEntity]) -> DetBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities)
        return DetBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            bboxes=[entity.bboxes for entity in entities],
            labels=[entity.labels for entity in entities],
        )

    def pin_memory(self) -> DetBatchDataEntity:
        """Pin memory for member tensor variables."""
        super().pin_memory()
        self.bboxes = [tv_tensors.wrap(bbox.pin_memory(), like=bbox) for bbox in self.bboxes]
        self.labels = [label.pin_memory() for label in self.labels]
        return self


@dataclass
class DetBatchPredEntity(DetBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for detection task."""


@dataclass
class DetBatchPredEntityWithXAI(DetBatchDataEntity, OTXBatchPredEntityWithXAI):
    """Data entity to represent model output predictions for detection task with explanations."""
