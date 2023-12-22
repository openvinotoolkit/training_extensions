# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXClassificationDatasets."""

from __future__ import annotations

from typing import Callable

import torch
from datumaro import Image, Label
from datumaro.components.annotation import AnnotationType
from torch.nn import functional

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import (
    MulticlassClsBatchDataEntity,
    MulticlassClsDataEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsDataEntity,
    HlabelClsBatchDataEntity,
    HlabelClsDataEntity,
    HLabelInfo
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
                pad_shape=img_shape,
                scale_factor=(1.0, 1.0),
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
                pad_shape=img_shape,
                scale_factor=(1.0, 1.0),
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
        hlabel_info = self._get_hlabel_info()
        breakpoint()

    def _get_item_impl(self, index: int) -> HlabelClsDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        label_anns = [ann for ann in item.annotations if isinstance(ann, Label)]

        entity = HlabelClsDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                pad_shape=img_shape,
                scale_factor=(1.0, 1.0),
            ),
            labels=torch.as_tensor([ann.label for ann in label_anns]),
            label_groups=torch.as_tensor([ann.group for ann in label_anns])
        )

        return self._apply_transforms(entity)

    def _get_hlabel_info(self):
        """Get H-label information will be used at the ModelAPI side."""
        label_groups = self.dm_subset.categories()[AnnotationType.label].label_groups
        
        num_multiclass_heads = 0
        num_multilabel_classes = 0
        multiclass_head_indices = []
        multilabel_head_indices = []
        for i, label_group in enumerate(label_groups):
            num_labels = len(label_group.labels)
            if num_labels > 1:
                num_multiclass_heads += 1
            else: 
                num_multilabel_classes += 1

    
    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect HlabelClsDataEntity into HlabelClsBatchDataEntity in data loader."""
        return HlabelClsBatchDataEntity.collate_fn
