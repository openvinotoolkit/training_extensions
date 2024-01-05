# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXClassificationDatasets."""

from __future__ import annotations

from operator import itemgetter
from typing import Any, Callable

import torch
from datumaro import Image, Label
from datumaro.components.annotation import AnnotationType
from torch.nn import functional

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

from .base import OTXDataset


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
            ),
            labels=torch.as_tensor([ann.label for ann in label_anns]),
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect MulticlassClsDataEntity into MulticlassClsBatchDataEntity in data loader."""
        return MulticlassClsBatchDataEntity.collate_fn


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
            ),
            labels=self._convert_to_onehot(labels),
        )

        return self._apply_transforms(entity)

    def _convert_to_onehot(self, labels: torch.tensor) -> torch.tensor:
        """Convert label to one-hot vector format."""
        return functional.one_hot(labels, self.num_classes).sum(0)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect MultilabelClsDataEntity into MultilabelClsBatchDataEntity in data loader."""
        return MultilabelClsBatchDataEntity.collate_fn


class OTXHlabelClsDataset(OTXDataset[HlabelClsDataEntity]):
    """OTXDataset class for H-label classification task."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dm_categories = self.dm_subset.categories()[AnnotationType.label]
        self.hlabel_info = self._get_hlabel_info()

        if self.hlabel_info.num_multiclass_heads == 0:
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
            ),
            labels=torch.as_tensor(hlabel_labels),
            hlabel_info=self.hlabel_info,
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
        num_multiclass_heads = self.hlabel_info.num_multiclass_heads
        num_multilabel_classes = self.hlabel_info.num_multilabel_classes

        # NOTE: currently ignored labels are not considered yet.
        ignored_labels: list = []

        class_indices = [0] * (num_multiclass_heads + num_multilabel_classes)
        for i in range(num_multiclass_heads):
            class_indices[i] = -1

        for ann in label_anns:
            ann_name = self.dm_categories.items[ann.label].name
            group_idx, in_group_idx = self.hlabel_info.class_to_group_idx[ann_name]

            if group_idx < num_multiclass_heads:
                class_indices[group_idx] = in_group_idx
            elif ann.label not in ignored_labels:
                class_indices[num_multiclass_heads + in_group_idx] = 1
            else:
                class_indices[num_multiclass_heads + in_group_idx] = -1

        return class_indices

    def _get_hlabel_info(self) -> HLabelInfo:
        """Get H-label information.

        To check the detailed information, please see the definition of HLabelInfo.
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
                "exclusive_groups": exclusive_groups,
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

        all_groups = [label_group.labels for label_group in self.dm_categories.label_groups]

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
            label_to_idx=self.dm_categories._indices,  # noqa: SLF001
            empty_multiclass_head_indices=[],  # consider the label removing case
        )

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect HlabelClsDataEntity into HlabelClsBatchDataEntity in data loader."""
        return HlabelClsBatchDataEntity.collate_fn
