# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX classification data entities."""

from __future__ import annotations

from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Any

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
    from datumaro import Label, LabelCategories
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
        stack_images: bool = True,
    ) -> MulticlassClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities)
        batch_images = (
            tv_tensors.Image(data=torch.stack(batch_data.images, dim=0)) if stack_images else batch_data.images
        )
        return MulticlassClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_images,
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
        stack_images: bool = True,
    ) -> MultilabelClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities)
        batch_images = (
            tv_tensors.Image(data=torch.stack(batch_data.images, dim=0)) if stack_images else batch_data.images
        )
        return MultilabelClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_images,
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

    @classmethod
    def from_dm_label_groups(cls, dm_label_categories: LabelCategories) -> HLabelInfo:
        """Generate HLabelInfo from the Datumaro LabelCategories.

        Args:
            dm_label_categories (LabelCategories): the label categories of datumaro.
        """

        def get_exclusive_group_info(all_groups: list[Label | list[Label]]) -> dict[str, Any]:
            """Get exclusive group information."""
            exclusive_groups = [g for g in all_groups if len(g) > 1]
            exclusive_groups.sort(key=itemgetter(0))

            last_logits_pos = 0
            num_single_label_classes = 0
            head_idx_to_logits_range = {}
            class_to_idx = {}

            for i, group in enumerate(exclusive_groups):
                head_idx_to_logits_range[str(i)] = (last_logits_pos, last_logits_pos + len(group))
                last_logits_pos += len(group)
                for j, c in enumerate(group):
                    class_to_idx[c] = (i, j)
                    num_single_label_classes += 1

            return {
                "num_multiclass_heads": len(exclusive_groups),
                "head_idx_to_logits_range": head_idx_to_logits_range,
                "class_to_idx": class_to_idx,
                "num_single_label_classes": num_single_label_classes,
            }

        def get_single_label_group_info(
            all_groups: list[Label | list[Label]],
            num_exclusive_groups: int,
        ) -> dict[str, Any]:
            """Get single label group information."""
            single_label_groups = [g for g in all_groups if len(g) == 1]
            single_label_groups.sort(key=itemgetter(0))

            class_to_idx = {}

            for i, group in enumerate(single_label_groups):
                class_to_idx[group[0]] = (num_exclusive_groups, i)

            return {
                "num_multilabel_classes": len(single_label_groups),
                "class_to_idx": class_to_idx,
            }

        def merge_class_to_idx(
            exclusive_ctoi: dict[str, tuple[int, int]],
            single_label_ctoi: dict[str, tuple[int, int]],
        ) -> dict[str, tuple[int, int]]:
            """Merge the class_to_idx information from exclusive and single_label groups."""

            def put_key_values(src: dict, dst: dict) -> None:
                """Put key and values from src to dst."""
                for k, v in src.items():
                    dst[k] = v

            class_to_idx: dict[str, tuple[int, int]] = {}
            put_key_values(exclusive_ctoi, class_to_idx)
            put_key_values(single_label_ctoi, class_to_idx)
            return class_to_idx

        all_groups = [label_group.labels for label_group in dm_label_categories.label_groups]

        exclusive_group_info = get_exclusive_group_info(all_groups)
        single_label_group_info = get_single_label_group_info(all_groups, exclusive_group_info["num_multiclass_heads"])

        merged_class_to_idx = merge_class_to_idx(
            exclusive_group_info["class_to_idx"],
            single_label_group_info["class_to_idx"],
        )

        return HLabelInfo(
            num_multiclass_heads=exclusive_group_info["num_multiclass_heads"],
            num_multilabel_classes=single_label_group_info["num_multilabel_classes"],
            head_idx_to_logits_range=exclusive_group_info["head_idx_to_logits_range"],
            num_single_label_classes=exclusive_group_info["num_single_label_classes"],
            class_to_group_idx=merged_class_to_idx,
            all_groups=all_groups,
            label_to_idx=dm_label_categories._indices,  # noqa: SLF001
            empty_multiclass_head_indices=[],  # consider the label removing case
        )


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
        batch_data = super().collate_fn(entities)
        batch_images = (
            tv_tensors.Image(data=torch.stack(batch_data.images, dim=0)) if stack_images else batch_data.images
        )
        return HlabelClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )


@dataclass
class HlabelClsBatchPredEntity(HlabelClsBatchDataEntity, OTXBatchPredEntity):
    """Data entity to represent model output predictions for H-label classification task."""
