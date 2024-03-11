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
class MulticlassClsDataEntity(OTXDataEntity):
    """Data entity for multi-class classification task.

    :param labels: labels as integer indices
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.MULTI_CLASS_CLS

    labels: LongTensor


@dataclass
class MulticlassClsPredEntity(MulticlassClsDataEntity, OTXPredEntity):
    """Data entity to represent the multi-class classification model output prediction."""


@dataclass
class MulticlassClsPredEntityWithXAI(MulticlassClsDataEntity, OTXPredEntityWithXAI):
    """Data entity to represent the multi-class classification model output prediction with explanations."""


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
        stack_images: bool = True,
    ) -> MulticlassClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        return MulticlassClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )

    def pin_memory(self) -> MulticlassClsBatchDataEntity:
        """Pin memory for member tensor variables."""
        super().pin_memory()
        self.labels = [label.pin_memory() for label in self.labels]
        return self


@dataclass
class MulticlassClsBatchPredEntity(MulticlassClsBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for multi-class classification task."""


@dataclass
class MulticlassClsBatchPredEntityWithXAI(MulticlassClsBatchDataEntity, OTXBatchPredEntityWithXAI):
    """Data entity to represent model output predictions for multi-class classification task with explanations."""


@register_pytree_node
@dataclass
class MultilabelClsDataEntity(OTXDataEntity):
    """Data entity for multi-label classification task.

    :param labels: Multi labels represented as an one-hot vector.
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.MULTI_LABEL_CLS

    labels: LongTensor


@dataclass
class MultilabelClsPredEntity(MultilabelClsDataEntity, OTXPredEntity):
    """Data entity to represent the multi-label classification model output prediction."""


@dataclass
class MultilabelClsPredEntityWithXAI(MultilabelClsDataEntity, OTXPredEntityWithXAI):
    """Data entity to represent the multi-label classification model output prediction with explanations."""


@dataclass
class MultilabelClsBatchDataEntity(OTXBatchDataEntity[MultilabelClsDataEntity]):
    """Data entity for multi-label classification task.

    :param labels: A list of labels as integer indices
    """

    labels: list[LongTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.MULTI_LABEL_CLS

    @classmethod
    def collate_fn(
        cls,
        entities: list[MultilabelClsDataEntity],
        stack_images: bool = True,
    ) -> MultilabelClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        return MultilabelClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )

    def pin_memory(self) -> MultilabelClsBatchDataEntity:
        """Pin memory for member tensor variables."""
        super().pin_memory()
        self.labels = [label.pin_memory() for label in self.labels]
        return self


@dataclass
class MultilabelClsBatchPredEntity(MultilabelClsBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for multi-label classification task."""


@dataclass
class MultilabelClsBatchPredEntityWithXAI(MultilabelClsBatchDataEntity, OTXBatchPredEntityWithXAI):
    """Data entity to represent model output predictions for multi-label classification task with explanations."""


@register_pytree_node
@dataclass
class HlabelClsDataEntity(OTXDataEntity):
    """Data entity for H-label classification task.

    :param labels: labels as integer indices
    :param label_group: the group of the label
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.H_LABEL_CLS

    labels: LongTensor


@dataclass
class HlabelClsPredEntity(HlabelClsDataEntity, OTXPredEntity):
    """Data entity to represent the H-label classification model output prediction."""


@dataclass
class HlabelClsPredEntityWithXAI(HlabelClsDataEntity, OTXPredEntityWithXAI):
    """Data entity to represent the H-label classification model output prediction with explanation."""


@dataclass
class HlabelClsBatchDataEntity(OTXBatchDataEntity[HlabelClsDataEntity]):
    """Data entity for H-label classification task.

    :param labels: A list of labels as integer indices
    :param label_groups: A list of label group
    """

    labels: list[LongTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.H_LABEL_CLS

    @classmethod
    def collate_fn(
        cls,
        entities: list[HlabelClsDataEntity],
        stack_images: bool = True,
    ) -> HlabelClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        return HlabelClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )


@dataclass
class HlabelClsBatchPredEntity(HlabelClsBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for H-label classification task."""


@dataclass
class HlabelClsBatchPredEntityWithXAI(HlabelClsBatchDataEntity, OTXBatchPredEntityWithXAI):
    """Data entity to represent model output predictions for H-label classification task with explanations."""
