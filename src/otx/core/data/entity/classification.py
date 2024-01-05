# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX classification data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
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
            images=tv_tensors.Image(data=torch.stack(batch_data.images, dim=0)),
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
    ) -> MultilabelClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities)
        return MultilabelClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=tv_tensors.Image(data=torch.stack(batch_data.images, dim=0)),
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
class HLabelInfo:
    """The label information represents the hierarchy.

    All params should be kept since they're also used at the Model API side.

    :param num_multiclass_heads: the number of the multiclass heads
    :param num_multilabel_classes: the number of multilabel classes
    :param head_to_logits_range: the logit range of each heads
    :param num_single_label_classes: the number of single label classes
    :param class_to_group_idx: represents the head index and label index
    :param all_groups: represents information of all groups
    :param label_to_idx: index of each label
    :param empty_multiclass_head_indices: the index of head that doesn't include any label
                                          due to the label removing

    i.e.
    Single-selection group information (Multiclass, Exclusive)
    {
        "Shape": ["Rigid", "Non-Rigid"],
        "Rigid": ["Rectangle", "Triangle"],
        "Non-Rigid": ["Circle"]
    }

    Multi-selection group information (Multilabel)
    {
        "Animal": ["Lion", "Panda"]
    }

    In the case above, HlabelInfo will be generated as below.
    NOTE, If there was only one label in the multiclass group, it will be handeled as multilabel(Circle).

        num_multiclass_heads: 2  (Shape, Rigid)
        num_multilabel_classes: 3 (Circle, Lion, Panda)
        head_to_logits_range: {'0': (0, 2), '1': (2, 4)} (Each multiclass head have 2 labels)
        num_single_label_classes: 4 (Rigid, Non-Rigid, Rectangle, Triangle)
        class_to_group_idx: {
            'Non-Rigid': (0, 0), 'Rigid': (0, 1),
            'Rectangle': (1, 0), 'Triangle': (1, 1),
            'Circle': (2, 0), 'Lion': (2,1), 'Panda': (2,2)
        } (head index, label index for each head)
        all_groups: [['Non-Rigid', 'Rigid'], ['Rectangle', 'Triangle'], ['Circle'], ['Lion'], ['Panda']]
        label_to_idx: {
            'Rigid': 0, 'Rectangle': 1,
            'Triangle': 2, 'Non-Rigid': 3, 'Circle': 4
            'Lion': 5, 'Panda': 6
        }
        empty_multiclass_head_indices: []

    All of the member variables should be considered for the Model API.
    https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/classification/utils/cls_utils.py#L97
    """

    num_multiclass_heads: int
    num_multilabel_classes: int
    head_idx_to_logits_range: dict[str, tuple[int, int]]
    num_single_label_classes: int
    class_to_group_idx: dict[str, tuple[int, int]]
    all_groups: list[list[str]]
    label_to_idx: dict[str, int]
    empty_multiclass_head_indices: list[int]


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
    hlabel_info: HLabelInfo


@dataclass
class HlabelClsPredEntity(HlabelClsDataEntity, OTXPredEntity):
    """Data entity to represent the H-label classification model output prediction."""


@dataclass
class HlabelClsBatchDataEntity(OTXBatchDataEntity[HlabelClsDataEntity]):
    """Data entity for H-label classification task.

    :param labels: A list of labels as integer indices
    :param label_groups: A list of label group
    """

    labels: list[LongTensor]
    hlabel_info: list[HLabelInfo]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.H_LABEL_CLS

    @classmethod
    def collate_fn(
        cls,
        entities: list[HlabelClsDataEntity],
    ) -> HlabelClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities)
        return HlabelClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
            hlabel_info=[entity.hlabel_info for entity in entities],
        )


@dataclass
class HlabelClsBatchPredEntity(HlabelClsBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for H-label classification task."""
