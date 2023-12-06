# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXDetectionDataset."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from datumaro import Bbox, DatasetSubset, Image
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetBatchDataEntity, DetDataEntity

from .base import OTXDataset, Transforms


class OTXDetectionDataset(OTXDataset[DetDataEntity]):
    """OTXDataset class for detection task."""

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        mem_cache_img_max_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            mem_cache_img_max_size=mem_cache_img_max_size,
        )

    def _get_item_impl(self, index: int) -> DetDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data = self._get_img_data(img)
        img_shape = img_data.shape[:2]

        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]

        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        entity = DetDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                pad_shape=img_shape,
                scale_factor=(1.0, 1.0),
            ),
            bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            labels=torch.as_tensor([ann.label for ann in bbox_anns]),
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect DetDataEntity into DetBatchDataEntity in data loader."""
        return DetBatchDataEntity.collate_fn
