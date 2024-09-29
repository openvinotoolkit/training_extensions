# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXClassificationDatasets."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from datumaro import Image, Label
from datumaro.components.annotation import AnnotationType
from torch.nn import functional

from otx.core.data.dataset.base import OTXDataset
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsDataEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsDataEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsDataEntity,
)
from otx.core.types.label import HLabelInfo


class OTXMulticlassClsDataset(OTXDataset[MulticlassClsDataEntity]):
    """OTXDataset class for multi-class classification task."""

    def _get_item_impl(self, index: int) -> MulticlassClsDataEntity | None:
        item = self.dm_subset[index]
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        label_anns = []
        for ann in item.annotations:
            if isinstance(ann, Label):
                label_anns.append(ann)
            else:
                # If the annotation is not Label, it should be converted to Label.
                # For Chained Task: Detection (Bbox) -> Classification (Label)
                label = Label(label=ann.label)
                if label not in label_anns:
                    label_anns.append(label)
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
        item = self.dm_subset[index]
        img = item.media_as(Image)
        ignored_labels: list[int] = []  # This should be assigned form item
        img_data, img_shape = self._get_img_data_and_shape(img)

        label_anns = []
        for ann in item.annotations:
            if isinstance(ann, Label):
                label_anns.append(ann)
            else:
                # If the annotation is not Label, it should be converted to Label.
                # For Chained Task: Detection (Bbox) -> Classification (Label)
                label = Label(label=ann.label)
                if label not in label_anns:
                    label_anns.append(label)
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
            parent_name = dm_label_category.parent if dm_label_category else ""

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
        item = self.dm_subset[index]
        img = item.media_as(Image)
        ignored_labels: list[int] = []  # This should be assigned form item
        img_data, img_shape = self._get_img_data_and_shape(img)

        label_anns = []
        for ann in item.annotations:
            if isinstance(ann, Label):
                label_anns.append(ann)
            else:
                # If the annotation is not Label, it should be converted to Label.
                # For Chained Task: Detection (Bbox) -> Classification (Label)
                label = Label(label=ann.label)
                if label not in label_anns:
                    label_anns.append(label)
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
        Total length of result is sum of number of hierarchy and number of multilabel classes.

        i.e.
        Let's assume that we used the same dataset with example of the definition of HLabelData
        and the original labels are ["Rigid", "Triangle", "Lion"].

        Then, h-label format will be [0, 1, 1, 0].
        The first N-th indices represent the label index of multiclass heads (N=num_multiclass_heads),
        others represent the multilabel labels.

        [Multiclass Heads]
        0-th index = 0 -> ["Rigid"(O), "Non-Rigid"(X)] <- First multiclass head
        1-st index = 1 -> ["Rectangle"(O), "Triangle"(X), "Circle"(X)] <- Second multiclass head

        [Multilabel Head]
        2, 3 indices = [1, 0] -> ["Lion"(O), "Panda"(X)]
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
            ann_parent = self.dm_categories.items[ann.label].parent
            group_idx, in_group_idx = self.label_info.class_to_group_idx[ann_name]
            (parent_group_idx, parent_in_group_idx) = (
                self.label_info.class_to_group_idx[ann_parent] if ann_parent else (None, None)
            )

            if group_idx < num_multiclass_heads:
                class_indices[group_idx] = in_group_idx
                if parent_group_idx is not None and parent_in_group_idx is not None:
                    class_indices[parent_group_idx] = parent_in_group_idx
            elif not ignored_labels or ann.label not in ignored_labels:
                class_indices[num_multiclass_heads + in_group_idx] = 1
            else:
                class_indices[num_multiclass_heads + in_group_idx] = -1

        return class_indices

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect HlabelClsDataEntity into HlabelClsBatchDataEntity in data loader."""
        return partial(HlabelClsBatchDataEntity.collate_fn, stack_images=self.stack_images)
