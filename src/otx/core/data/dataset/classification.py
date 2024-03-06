# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXClassificationDatasets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import torch
from datumaro import Image, Label
from datumaro.components.annotation import AnnotationType
from torch.nn import functional

from otx.core.data.dataset.base import LabelInfo, OTXDataset
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsDataEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsDataEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsDataEntity,
)

if TYPE_CHECKING:
    from datumaro import LabelCategories


@dataclass
class HLabelInfo(LabelInfo):
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
        label_tree_edges: [
            ["Rectangle", "Rigid"], ["Triangle", "Rigid"], ["Circle", "Non-Rigid"],
        ] # NOTE, label_tree_edges format could be changed.
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
    label_tree_edges: list[list[str]]
    empty_multiclass_head_indices: list[int]

    @classmethod
    def from_dm_label_groups(cls, dm_label_categories: LabelCategories) -> HLabelInfo:
        """Generate HLabelData from the Datumaro LabelCategories.

        Args:
            dm_label_categories (LabelCategories): the label categories of datumaro.
        """

        def get_exclusive_group_info(all_groups: list[Label | list[Label]]) -> dict[str, Any]:
            """Get exclusive group information."""
            exclusive_groups = [g for g in all_groups if len(g) > 1]

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

        def get_label_tree_edges(dm_label_items: list[LabelCategories]) -> list[list[str]]:
            """Get label tree edges information. Each edges represent [child, parent]."""
            return [[item.name, item.parent] for item in dm_label_items if item.parent != ""]

        all_groups = [label_group.labels for label_group in dm_label_categories.label_groups]

        exclusive_group_info = get_exclusive_group_info(all_groups)
        single_label_group_info = get_single_label_group_info(all_groups, exclusive_group_info["num_multiclass_heads"])

        merged_class_to_idx = merge_class_to_idx(
            exclusive_group_info["class_to_idx"],
            single_label_group_info["class_to_idx"],
        )

        return HLabelInfo(
            label_names=[item.name for item in dm_label_categories.items],
            label_groups=all_groups,
            num_multiclass_heads=exclusive_group_info["num_multiclass_heads"],
            num_multilabel_classes=single_label_group_info["num_multilabel_classes"],
            head_idx_to_logits_range=exclusive_group_info["head_idx_to_logits_range"],
            num_single_label_classes=exclusive_group_info["num_single_label_classes"],
            class_to_group_idx=merged_class_to_idx,
            all_groups=all_groups,
            label_to_idx=dm_label_categories._indices,  # noqa: SLF001
            label_tree_edges=get_label_tree_edges(dm_label_categories.items),
            empty_multiclass_head_indices=[],  # consider the label removing case
        )


class OTXMulticlassClsDataset(OTXDataset[MulticlassClsDataEntity]):
    """OTXDataset class for multi-class classification task."""

    def _get_item_impl(self, index: int) -> MulticlassClsDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        label_anns = [ann for ann in item.annotations if isinstance(ann, Label)]
        if len(label_anns) > 1:
            msg = f"Multi-class Classification can't use the multi-label, currently len(labels) = {len(label_anns)}"
            raise ValueError(msg)

        entity = MulticlassClsDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
            ),
            labels=torch.as_tensor([ann.label for ann in label_anns]),
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect MulticlassClsDataEntity into MulticlassClsBatchDataEntity in data loader."""
        return partial(MulticlassClsBatchDataEntity.collate_fn, stack_images=self.stack_images)


class OTXMultilabelClsDataset(OTXDataset[MultilabelClsDataEntity]):
    """OTXDataset class for multi-label classification task."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = len(self.dm_subset.categories()[AnnotationType.label])

    def _get_item_impl(self, index: int) -> MultilabelClsDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        ignored_labels: list[int] = []  # This should be assigned form item
        img_data, img_shape = self._get_img_data_and_shape(img)

        label_anns = [ann for ann in item.annotations if isinstance(ann, Label)]
        labels = torch.as_tensor([ann.label for ann in label_anns])

        entity = MultilabelClsDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=ignored_labels,
            ),
            labels=self._convert_to_onehot(labels, ignored_labels),
        )

        return self._apply_transforms(entity)

    def _convert_to_onehot(self, labels: torch.tensor, ignored_labels: list[int]) -> torch.tensor:
        """Convert label to one-hot vector format."""
        onehot = functional.one_hot(labels, self.num_classes).sum(0).clamp_max_(1)
        if ignored_labels:
            for ignore_label in ignored_labels:
                onehot[ignore_label] = -1
        return onehot

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect MultilabelClsDataEntity into MultilabelClsBatchDataEntity in data loader."""
        return partial(MultilabelClsBatchDataEntity.collate_fn, stack_images=self.stack_images)


class OTXHlabelClsDataset(OTXDataset[HlabelClsDataEntity]):
    """OTXDataset class for H-label classification task."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dm_categories = self.dm_subset.categories()[AnnotationType.label]

        # Hlabel classification used HLabelInfo to insert the HLabelData.
        self.label_info = HLabelInfo.from_dm_label_groups(self.dm_categories)
        if self.label_info.num_multiclass_heads == 0:
            msg = "The number of multiclass heads should be larger than 0."
            raise ValueError(msg)

        for dm_item in self.dm_subset:
            self._add_ancestors(dm_item.annotations)

    def _add_ancestors(self, label_anns: list[Label]) -> None:
        """Add ancestors recursively if some label miss the ancestor information.

        If the label tree likes below,
        object - vehicle -- car
                         |- bus
                         |- truck
        And annotation = ['car'], it should be ['car', 'vehicle', 'object'], to include the ancestor.

        This function add the ancestors to the annotation if missing.
        """

        def _label_idx_to_name(idx: int) -> str:
            return self.label_info.label_names[idx]

        def _label_name_to_idx(name: str) -> int:
            indices = [idx for idx, val in enumerate(self.label_info.label_names) if val == name]
            return indices[0]

        def _get_label_group_idx(label_name: str) -> int:
            if isinstance(self.label_info, HLabelInfo):
                return self.label_info.class_to_group_idx[label_name][0]
            msg = f"self.label_info should have HLabelInfo type, got {type(self.label_info)}"
            raise ValueError(msg)

        def _find_ancestor_recursively(label_name: str, ancestors: list) -> list[str]:
            _, dm_label_category = self.dm_categories.find(label_name)
            parent_name = dm_label_category.parent

            if parent_name != "":
                ancestors.append(parent_name)
                _find_ancestor_recursively(parent_name, ancestors)
            return ancestors

        def _get_all_label_names_in_anns(anns: list[Label]) -> list[str]:
            return [_label_idx_to_name(ann.label) for ann in anns]

        all_label_names = _get_all_label_names_in_anns(label_anns)
        ancestor_dm_labels = []
        for ann in label_anns:
            label_idx = ann.label
            label_name = _label_idx_to_name(label_idx)
            ancestors = _find_ancestor_recursively(label_name, [])

            for i, ancestor in enumerate(ancestors):
                if ancestor not in all_label_names:
                    ancestor_dm_labels.append(
                        Label(
                            label=_label_name_to_idx(ancestor),
                            id=len(label_anns) + i,
                            group=_get_label_group_idx(ancestor),
                        ),
                    )
        label_anns.extend(ancestor_dm_labels)

    def _get_item_impl(self, index: int) -> HlabelClsDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        ignored_labels: list[int] = []  # This should be assigned form item
        img_data, img_shape = self._get_img_data_and_shape(img)

        label_anns = [ann for ann in item.annotations if isinstance(ann, Label)]
        self._add_ancestors(label_anns)
        hlabel_labels = self._convert_label_to_hlabel_format(label_anns, ignored_labels)

        entity = HlabelClsDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=ignored_labels,
            ),
            labels=torch.as_tensor(hlabel_labels),
        )

        return self._apply_transforms(entity)

    def _convert_label_to_hlabel_format(self, label_anns: list[Label], ignored_labels: list[int]) -> list[int]:
        """Convert format of the label to the h-label.

        It converts the label format to h-label format.

        i.e.
        Let's assume that we used the same dataset with example of the definition of HLabelData
        and the original labels are ["Rigid", "Panda", "Lion"].

        Then, h-label format will be [1, -1, 0, 1, 1].
        The first N-th indices represent the label index of multiclass heads (N=num_multiclass_heads),
        others represent the multilabel labels.

        [Multiclass Heads: [1, -1]]
        0-th index = 1 -> ["Non-Rigid"(X), "Rigid"(O)] <- First multiclass head
        1-st index = -1 -> ["Rectangle"(X), "Triangle"(X)] <- Second multiclass head

        [Multilabel Head: [0, 1, 1]]
        2, 3, 4 indices = [0, 1, 1] -> ["Circle"(X), "Lion"(O), "Panda"(O)]
        """
        if not isinstance(self.label_info, HLabelInfo):
            msg = f"The type of label_info should be HLabelInfo, got {type(self.label_info)}."
            raise TypeError(msg)

        num_multiclass_heads = self.label_info.num_multiclass_heads
        num_multilabel_classes = self.label_info.num_multilabel_classes

        class_indices = [0] * (num_multiclass_heads + num_multilabel_classes)
        for i in range(num_multiclass_heads):
            class_indices[i] = -1

        for ann in label_anns:
            ann_name = self.dm_categories.items[ann.label].name
            group_idx, in_group_idx = self.label_info.class_to_group_idx[ann_name]

            if group_idx < num_multiclass_heads:
                class_indices[group_idx] = in_group_idx
            elif not ignored_labels or ann.label not in ignored_labels:
                class_indices[num_multiclass_heads + in_group_idx] = 1
            else:
                class_indices[num_multiclass_heads + in_group_idx] = -1

        return class_indices

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect HlabelClsDataEntity into HlabelClsBatchDataEntity in data loader."""
        return partial(HlabelClsBatchDataEntity.collate_fn, stack_images=self.stack_images)
