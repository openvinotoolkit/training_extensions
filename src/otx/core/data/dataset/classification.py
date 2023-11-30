# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXClassificationDataset."""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import torch
from datumaro import DatasetSubset, Image, Label

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import MulticlassClsBatchDataEntity, MulticlassClsDataEntity

from .base import OTXDataset, Transforms


class OTXMulticlassClsDataset(OTXDataset[MulticlassClsDataEntity]):
    """OTXDataset class for multi-class classification task."""

    def __init__(self, dm_subset: DatasetSubset, transforms: Transforms) -> None:
        super().__init__(dm_subset, transforms)

    def _get_item_impl(self, index: int) -> Optional[MulticlassClsDataEntity]:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)

        img = item.media_as(Image)
        img_data = img.data
        # TODO(vinnamkim): This is a temporal approach
        # There is an upcoming Datumaro patch here for this
        # https://github.com/openvinotoolkit/datumaro/pull/1194
        if img_data.shape[-1] == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        img_shape = img.size

        label_anns = [ann for ann in item.annotations if isinstance(ann, Label)]

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
