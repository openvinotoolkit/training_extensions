# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXSegmentationDataset."""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
import torch
from datumaro import Mask, DatasetSubset, Image
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegDataEntity

from .base import OTXDataset, Transforms

class OTXSegmentationDataset(OTXDataset[SegDataEntity]):
    """OTXDataset class for segmentation task."""

    def _get_item_impl(self, index: int) -> Optional[SegDataEntity]:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)

        img = item.media_as(Image)
        img_data = img.data

        if img_data.shape[-1] == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        img_shape = img.size

        mask_anns = [ann for ann in item.annotations if isinstance(ann, Mask)]

        breakpoint()
        masks = (
            np.stack([ann.points for ann in mask_anns], axis=0).astype(np.float32)
            if len(mask_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        entity = SegDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                pad_shape=img_shape,
                scale_factor=(1.0, 1.0),
            ),
            bboxes=tv_tensors.Mask(
                masks
            )
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect SegDataEntity into SegBatchDataEntity in data loader."""
        return SegBatchDataEntity.collate_fn
