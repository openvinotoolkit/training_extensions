# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX action data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from otx.core.data.entity.base import (
    OTXBatchDataEntity,
    OTXBatchPredEntity,
    OTXDataEntity,
    OTXPredEntity,
)
from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from torch import LongTensor
    from torchvision import tv_tensors


@register_pytree_node
@dataclass
class ActionDetDataEntity(OTXDataEntity):
    """Data entity for action classification task.

    Args:
        bboxes: 2D bounding boxes for actors.
        labels: Video's action labels.
    """

    bboxes: tv_tensors.BoundingBoxes
    labels: LongTensor
    frame_path: str | None = None
    proposal_file: str | None = None
    proposals: tv_tensors.BoundingBoxes | None = None

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.ACTION_DETECTION


@dataclass
class ActionDetPredEntity(ActionDetDataEntity, OTXPredEntity):
    """Data entity to represent the action classification model's output prediction."""


@dataclass
class ActionDetBatchDataEntity(OTXBatchDataEntity[ActionDetDataEntity]):
    """Batch data entity for action classification.

    Args:
        bboxes(list[tv_tensors.BoundingBoxes]): A list of bounding boxes of videos.
        labels(list[LongTensor]): A list of labels of videos.
    """

    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[LongTensor]
    proposals: list[tv_tensors.BoundingBoxes | None] | None = None

    @property
    def task(self) -> OTXTaskType:
        """OTX task type definition."""
        return OTXTaskType.ACTION_DETECTION

    @classmethod
    def collate_fn(cls, entities: list[ActionDetDataEntity]) -> ActionDetBatchDataEntity:
        """Collection function to collect `ActionClsDataEntity` into `ActionClsBatchDataEntity`."""
        batch_data = super().collate_fn(entities)
        return ActionDetBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            bboxes=[entity.bboxes for entity in entities],
            labels=[entity.labels for entity in entities],
            proposals=[entity.proposals for entity in entities],
        )


@dataclass
class ActionDetBatchPredEntity(ActionDetBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for action classification task."""
