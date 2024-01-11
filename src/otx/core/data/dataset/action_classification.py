# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXActionClsDataset."""

from __future__ import annotations

from typing import Callable

import torch
from datumaro import Label

from otx.core.data.dataset.base import OTXDataset
from otx.core.data.entity.action_classification import ActionClsBatchDataEntity, ActionClsDataEntity
from otx.core.data.entity.base import ImageInfo


class OTXActionClsDataset(OTXDataset[ActionClsDataEntity]):
    """OTXDataset class for action classification task."""

    def _get_item_impl(self, idx: int) -> ActionClsDataEntity | None:
        item = self.dm_subset.get(id=self.ids[idx], subset=self.dm_subset.name)

        label_anns = [ann for ann in item.annotations if isinstance(ann, Label)]

        entity = ActionClsDataEntity(
            video=item.media,
            image=[],
            img_info=ImageInfo(
                img_idx=idx,
                img_shape=(0, 0),
                ori_shape=(0, 0),
                attributes={},
            ),
            labels=torch.as_tensor([ann.label for ann in label_anns]),
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect ActionClsDataEntity into ActionClsBatchDataEntity."""
        return ActionClsBatchDataEntity.collate_fn
