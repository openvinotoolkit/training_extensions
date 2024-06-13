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
    VideoInfo,
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
    video_info: VideoInfo
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
class ActionClsPredEntity(OTXPredEntity, ActionClsDataEntity):
    """Data entity to represent the action classification model's output prediction."""


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
    def collate_fn(
        cls,
        entities: list[ActionClsDataEntity],
        stack_images: bool = True,
    ) -> ActionClsBatchDataEntity:
        """Collection function to collect `ActionClsDataEntity` into `ActionClsBatchDataEntity`."""
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        return ActionClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )

    def pin_memory(self) -> ActionClsBatchDataEntity:
        """Pin memory for member tensor variables."""
        return super().pin_memory().wrap(labels=[label.pin_memory() for label in self.labels])


@dataclass
class ActionClsBatchPredEntity(OTXBatchPredEntity, ActionClsBatchDataEntity):
    """Data entity to represent model output predictions for action classification task."""
