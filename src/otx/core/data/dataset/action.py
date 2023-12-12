# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXActionDataset."""

from __future__ import annotations

from typing import Callable

import torch
from datumaro import Image, Label

from otx.core.data.dataset.base import OTXDataset
from otx.core.data.entity.action import ActionClsBatchDataEntity, ActionClsDataEntity
from otx.core.data.entity.base import ImageInfo


class OTXActionClsDataet(OTXDataset[ActionClsDataEntity]):
    """OTXDataset class for action classification task."""

    def _get_item_impl(self, index: int) -> ActionClsDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        label_anns = [ann for ann in item.annotations if isinstance(ann, Label)]

        entity = ActionClsDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                pad_shape=img_shape,
                scale_factor=(1.0, 1.0),
            ),
            labels=torch.as_tensor(ann.label for ann in label_anns),
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect ActionClsDataEntity into ActionClsBatchDataEntity."""
        return ActionClsBatchDataEntity.collate_fn
