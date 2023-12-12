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


@register_pytree_node
@dataclass
class ActionClsDataEntity(OTXDataEntity):
    """Data entity for action classification task.

    Args:
        labels: Video's action labels.
    """

    labels: LongTensor

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.ACTION_CLASSIFICATION


@dataclass
class ActionClsPredEntity(ActionClsDataEntity, OTXPredEntity):
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
    def collate_fn(cls, entities: list[ActionClsDataEntity]) -> ActionClsBatchDataEntity:
        """Collection function to collect `ActionClsDataEntity` into `ActionClsBatchDataEntity`."""
        batch_data = super().collate_fn(entities)
        return ActionClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )


@dataclass
class ActionClsBatchPredEntity(ActionClsBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for action classification task."""
