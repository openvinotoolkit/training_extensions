# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX classification data entities."""

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
class MulticlassClsDataEntity(OTXDataEntity):
    """Data entity for multi-class classification task.

    :param labels: Bbox labels as integer indices
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.MULTI_CLASS_CLS

    labels: LongTensor


@dataclass
class MulticlassClsPredEntity(MulticlassClsDataEntity, OTXPredEntity):
    """Data entity to represent the detection model output prediction."""


@dataclass
class MulticlassClsBatchDataEntity(OTXBatchDataEntity[MulticlassClsDataEntity]):
    """Data entity for multi-class classification task.

    :param labels: A list of bbox labels as integer indices
    """

    labels: list[LongTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.MULTI_CLASS_CLS

    @classmethod
    def collate_fn(
        cls,
        entities: list[MulticlassClsDataEntity],
    ) -> MulticlassClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities)
        return MulticlassClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )


@dataclass
class MulticlassClsBatchPredEntity(MulticlassClsBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for multi-class classification task."""


@dataclass
class LabelGroup:
    """The label group represents the hierarchy.
    
    :param group_name: the name of the label group
    :param labels: labels included in the group
    """
    group_name: str
    labels: list[int]


@register_pytree_node
@dataclass
class HlabelClsDataEntity(OTXDataEntity):
    """Data entity for H-label classification task.

    :param labels: labels as integer indices
    :param label_groups: the list of label group
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.H_LABEL_CLS

    labels: LongTensor
    label_groups: list[LabelGroup]


@dataclass
class HlabelClsPredEntity(HlabelClsDataEntity, OTXPredEntity):
    """Data entity to represent the h-label classification model output prediction."""


@dataclass
class HlabelClsBatchDataEntity(OTXBatchDataEntity[HlabelClsDataEntity]):
    """Batch Data entity for h-label classification task.

    :param labels: A list of labels as integer indices
    :param label_groups: A list of label groups
    """

    labels: list[LongTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.MULTI_CLASS_CLS

    @classmethod
    def collate_fn(
        cls,
        entities: list[MulticlassClsDataEntity],
    ) -> MulticlassClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities)
        return MulticlassClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )


@dataclass
class MulticlassClsBatchPredEntity(MulticlassClsBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for multi-class classification task."""
