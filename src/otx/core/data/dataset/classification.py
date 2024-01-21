# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXClassificationDatasets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
from datumaro import Image, Label
from datumaro.components.annotation import AnnotationType
from torch.nn import functional

from otx.core.data.dataset.base import LabelInfo, OTXDataset
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsDataEntity,
    HLabelInfo,
    MulticlassClsBatchDataEntity,
    MulticlassClsDataEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsDataEntity,
)


@dataclass
class HLabelMetaInfo(LabelInfo):
    """Meta information of hlabel classification."""

    hlabel_info: HLabelInfo


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
            ),
            labels=self._convert_to_onehot(labels),
        )

        return self._apply_transforms(entity)

    def _convert_to_onehot(self, labels: torch.tensor) -> torch.tensor:
        """Convert label to one-hot vector format."""
        return functional.one_hot(labels, self.num_classes).sum(0).clamp_max_(1)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect MultilabelClsDataEntity into MultilabelClsBatchDataEntity in data loader."""
        return partial(MultilabelClsBatchDataEntity.collate_fn, stack_images=self.stack_images)


class OTXHlabelClsDataset(OTXDataset[HlabelClsDataEntity]):
    """OTXDataset class for H-label classification task."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dm_categories = self.dm_subset.categories()[AnnotationType.label]

        # Hlabel classification used HLabelMetaInfo to insert the HLabelInfo.
        self.meta_info = HLabelMetaInfo(
            label_names=[category.name for category in self.dm_categories],
            hlabel_info=HLabelInfo.from_dm_label_groups(self.dm_categories),
        )

        if self.meta_info.hlabel_info.num_multiclass_heads == 0:
            msg = "The number of multiclass heads should be larger than 0."
            raise ValueError(msg)

    def _get_item_impl(self, index: int) -> HlabelClsDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        label_anns = [ann for ann in item.annotations if isinstance(ann, Label)]
        hlabel_labels = self._convert_label_to_hlabel_format(label_anns)

        entity = HlabelClsDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
            ),
            labels=torch.as_tensor(hlabel_labels),
        )

        return self._apply_transforms(entity)

    def _convert_label_to_hlabel_format(self, label_anns: list[Label]) -> list[int]:
        """Convert format of the label to the h-label.

        It converts the label format to h-label format.

        i.e.
        Let's assume that we used the same dataset with example of the definition of HLabelInfo
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
        if not isinstance(self.meta_info, HLabelMetaInfo):
            msg = f"The type of meta_info should be HLabelMetaInfo, got {type(self.meta_info)}."
            raise TypeError(msg)

        num_multiclass_heads = self.meta_info.hlabel_info.num_multiclass_heads
        num_multilabel_classes = self.meta_info.hlabel_info.num_multilabel_classes

        # NOTE: currently ignored labels are not considered yet.
        ignored_labels: list = []

        class_indices = [0] * (num_multiclass_heads + num_multilabel_classes)
        for i in range(num_multiclass_heads):
            class_indices[i] = -1

        for ann in label_anns:
            ann_name = self.dm_categories.items[ann.label].name
            group_idx, in_group_idx = self.meta_info.hlabel_info.class_to_group_idx[ann_name]

            if group_idx < num_multiclass_heads:
                class_indices[group_idx] = in_group_idx
            elif ann.label not in ignored_labels:
                class_indices[num_multiclass_heads + in_group_idx] = 1
            else:
                class_indices[num_multiclass_heads + in_group_idx] = -1

        return class_indices

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect HlabelClsDataEntity into HlabelClsBatchDataEntity in data loader."""
        return partial(HlabelClsBatchDataEntity.collate_fn, stack_images=self.stack_images)
