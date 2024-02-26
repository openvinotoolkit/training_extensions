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
    OTXBatchPredEntityWithXAI,
    OTXDataEntity,
    OTXPredEntity,
    OTXPredEntityWithXAI,
)
from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from datumaro.components.media import Video
    from torch import LongTensor


@register_pytree_node
@dataclass
class ActionClsDataEntity(OTXDataEntity):
    """Data entity for action classification task.

    Args:
        video: Video object.
        labels: Video's action labels.
    """

    video: Video
    labels: LongTensor

    def to_tv_image(self) -> ActionClsDataEntity:
        """Convert `self.image` to TorchVision Image if it is a Numpy array (inplace operation).

        Action classification data do not have image, so this will return itself.
        """
        return self

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.ACTION_CLASSIFICATION


@dataclass
class ActionClsPredEntity(ActionClsDataEntity, OTXPredEntity):
    """Data entity to represent the action classification model's output prediction."""


@dataclass
class ActionClsPredEntityWithXAI(ActionClsDataEntity, OTXPredEntityWithXAI):
    """Data entity to represent the detection model output prediction with explanations."""


@dataclass
class ActionClsBatchDataEntity(OTXBatchDataEntity[ActionClsDataEntity]):
    """Batch data entity for action classification.

    Args:
        labels(list[LongTensor]): A list of labels of videos.
    """

    labels: list[LongTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX task type definition."""
        return OTXTaskType.ACTION_CLASSIFICATION

    @classmethod
    def collate_fn(cls, entities: list[ActionClsDataEntity]) -> ActionClsBatchDataEntity:
        """Collection function to collect `ActionClsDataEntity` into `ActionClsBatchDataEntity`."""
        batch_data = super().collate_fn(entities)
        return ActionClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )

    def pin_memory(self) -> ActionClsBatchDataEntity:
        """Pin memory for member tensor variables."""
        super().pin_memory()
        self.labels = [label.pin_memory() for label in self.labels]
        return self


@dataclass
class ActionClsBatchPredEntity(ActionClsBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for action classification task."""


@dataclass
class ActionClsBatchPredEntityWithXAI(ActionClsBatchDataEntity, OTXBatchPredEntityWithXAI):
    """Data entity to represent model output predictions for multi-class classification task with explanations."""
