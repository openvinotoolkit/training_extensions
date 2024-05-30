# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXInstanceSegDataset."""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
import torch
from datumaro import Dataset as DmDataset
from datumaro import Image, Polygon
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegDataEntity
from otx.core.utils.mask_util import polygon_to_bitmap

from .base import OTXDataset, Transforms


class OTXInstanceSegDataset(OTXDataset[InstanceSegDataEntity]):
    """OTXDataset class for instance segmentation.

    Args:
        dm_subset (DmDataset): The subset of the dataset.
        transforms (Transforms): Data transformations to be applied.
        include_polygons (bool): Flag indicating whether to include polygons in the dataset.
            If set to False, polygons will be converted to bitmaps, and bitmaps will be used for training.
        **kwargs: Additional keyword arguments passed to the base class.
    """

    def __init__(self, dm_subset: DmDataset, transforms: Transforms, include_polygons: bool, **kwargs) -> None:
        super().__init__(dm_subset, transforms, **kwargs)
        self.include_polygons = include_polygons

    def _get_item_impl(self, index: int) -> InstanceSegDataEntity | None:
        item = self.dm_subset[index]
        img = item.media_as(Image)
        ignored_labels: list[int] = []
        img_data, img_shape = self._get_img_data_and_shape(img)

        gt_bboxes, gt_labels, gt_masks, gt_polygons = [], [], [], []

        for annotation in item.annotations:
            if isinstance(annotation, Polygon):
                bbox = np.array(annotation.get_bbox(), dtype=np.float32)
                gt_bboxes.append(bbox)
                gt_labels.append(annotation.label)

                if self.include_polygons:
                    gt_polygons.append(annotation)
                else:
                    gt_masks.append(polygon_to_bitmap([annotation], *img_shape)[0])

        # convert xywh to xyxy format
        bboxes = np.array(gt_bboxes, dtype=np.float32) if gt_bboxes else np.empty((0, 4))
        bboxes[:, 2:] += bboxes[:, :2]

        masks = np.stack(gt_masks, axis=0) if gt_masks else np.zeros((0, *img_shape), dtype=bool)
        labels = np.array(gt_labels, dtype=np.int64)

        entity = InstanceSegDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=ignored_labels,
            ),
            bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            masks=tv_tensors.Mask(masks, dtype=torch.uint8),
            labels=torch.as_tensor(labels),
            polygons=gt_polygons,
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect InstanceSegDataEntity into InstanceSegDataEntity in dataloader."""
        return partial(InstanceSegBatchDataEntity.collate_fn, stack_images=self.stack_images)
