# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXDetectionDataset."""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
import torch
from datumaro import DatasetSubset, Image, Polygon
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegDataEntity

from .base import OTXDataset, Transforms


class OTXInstanceSegDataset(OTXDataset[InstanceSegDataEntity]):
    """OTXDataset class for detection task."""

    def __init__(self, dm_subset: DatasetSubset, transforms: Transforms) -> None:
        super().__init__(dm_subset, transforms)

    def _get_item_impl(self, index: int) -> Optional[InstanceSegDataEntity]:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)

        img = item.media_as(Image)
        img_data = img.data
        if img_data.shape[-1] == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        img_shape = img.size

        gt_bboxes = np.zeros(shape=(0, 4), dtype=np.float32)
        gt_labels = np.zeros(shape=(0, ), dtype=int)
        gt_masks = np.zeros(shape=(0, *img_shape), dtype=bool)
        for annotation in item.annotations:
            if isinstance(annotation, Polygon):
                points = np.array(annotation.points).reshape(-1, 2).astype(np.int32)
                x1, y1 = np.min(points, axis=0)
                x2, y2 = np.max(points, axis=0)
                gt_bboxes = np.vstack((gt_bboxes, np.array([x1, y1, x2, y2])))

                mask = np.zeros(img_shape)
                cv2.fillPoly(mask, [points], 1)
                gt_masks = np.vstack((gt_masks, mask[np.newaxis]))
                gt_labels = np.append(gt_labels, annotation.label)

        entity = InstanceSegDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                pad_shape=img_shape,
                scale_factor=(1.0, 1.0),
            ),
            bboxes=tv_tensors.BoundingBoxes(
                gt_bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            masks=tv_tensors.Mask(gt_masks, dtype=torch.uint8),
            labels=torch.as_tensor(gt_labels),
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect DetDataEntity into InstanceSegDataEntity in data loader."""
        return InstanceSegBatchDataEntity.collate_fn
