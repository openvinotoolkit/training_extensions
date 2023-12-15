# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXActionDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import cv2
import torch
from datumaro import Label

if TYPE_CHECKING:
    import numpy as np
    from datumaro.components.dataset_base import DatasetItem

from otx.core.data.dataset.base import OTXDataset
from otx.core.data.entity.action import ActionClsBatchDataEntity, ActionClsDataEntity
from otx.core.data.entity.base import ImageInfo


class OTXActionClsDataset(OTXDataset[ActionClsDataEntity]):
    """OTXDataset class for action classification task."""

    def _get_item_impl(self, idx: int) -> ActionClsDataEntity | None:
        item = self.dm_subset.get(id=self.ids[idx], subset=self.dm_subset.name)
        # video_data = [frame.data for frame in video]
        video_data = self._get_video_data(item)
        img_shape = video_data[0].shape

        label_anns = [ann for ann in item.annotations if isinstance(ann, Label)]

        entity = ActionClsDataEntity(
            image=video_data,
            img_info=ImageInfo(
                img_idx=idx,
                img_shape=img_shape,
                ori_shape=img_shape,
                pad_shape=img_shape,
                scale_factor=(1.0, 1.0),
            ),
            labels=torch.as_tensor([ann.label for ann in label_anns]),
        )

        return self._apply_transforms(entity)

    @staticmethod
    def _get_video_data(item: DatasetItem) -> list[np.ndarray]:
        """Return frames from datumaro dataset video item.

        This can be handled easily by [frame.data for frame in video],
        but latest Datumaro raises error after some iterateions.
        Therefore, this is temporary workaround to get frames.
        """
        video = cv2.VideoCapture(item.media.path)
        video_data = []
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                video_data.append(frame)
            else:
                break
        video.release()
        return video_data

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect ActionClsDataEntity into ActionClsBatchDataEntity."""
        return ActionClsBatchDataEntity.collate_fn
