# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX action data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchvision import tv_tensors

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


@register_pytree_node
@dataclass
class ActionDetDataEntity(OTXDataEntity):
    """Data entity for action classification task.

    Args:
        bboxes: 2D bounding boxes for actors.
        labels: One-hot vector of video's action labels.
        frame_path: Data media's file path for getting proper meta information.
        proposals: Pre-calculated actor proposals.
    """

    bboxes: tv_tensors.BoundingBoxes
    labels: LongTensor
    frame_path: str
    proposals: tv_tensors.BoundingBoxes

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
    proposals: list[tv_tensors.BoundingBoxes]

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

    def pin_memory(self) -> ActionDetBatchDataEntity:
        """Pin memory for member tensor variables."""
        super().pin_memory()
        self.bboxes = [tv_tensors.wrap(bbox.pin_memory(), like=bbox) for bbox in self.bboxes]
        self.labels = [label.pin_memory() for label in self.labels]
        self.proposals = [tv_tensors.wrap(proposal.pin_memory(), like=proposal) for proposal in self.proposals]
        return self


@dataclass
class ActionDetBatchPredEntity(ActionDetBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for action classification task."""
