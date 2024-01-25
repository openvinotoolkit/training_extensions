# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXSegmentationDataset."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from datumaro.components.annotation import Image, Mask
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegDataEntity

from .base import OTXDataset


class OTXSegmentationDataset(OTXDataset[SegDataEntity]):
    """OTXDataset class for segmentation task."""

    def _get_item_impl(self, index: int) -> SegDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        ignored_labels: list | None = None  # This should be assigned form item
        img_data, img_shape = self._get_img_data_and_shape(img)

        # create 2D class mask. We use np.sum() since Datumaro returns 3D masks (one for each class)
        mask_anns = np.sum(
            [ann.as_class_mask() for ann in item.annotations if isinstance(ann, Mask)],
            axis=0,
            dtype=np.uint8,
        )

        entity = SegDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=ignored_labels,
            ),
            gt_seg_map=tv_tensors.Mask(
                torch.as_tensor(mask_anns, dtype=torch.long),
            ),
        )
        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect SegDataEntity into SegBatchDataEntity in data loader."""
        return SegBatchDataEntity.collate_fn
