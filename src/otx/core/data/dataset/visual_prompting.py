# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXVisualPromptingDataset."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from datumaro import DatasetSubset, Image, Polygon
from torchvision import tv_tensors
import torchvision.transforms.v2 as tvt_v2

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity, VisualPromptingDataEntity)
from otx.core.utils.mask_util import polygon_to_bitmap

from .base import OTXDataset, Transforms


class OTXVisualPromptingDataset(OTXDataset[VisualPromptingDataEntity]):
    """OTXDataset class for visual prompting.

    Args:
        dm_subset (DatasetSubset): The subset of the dataset.
        transforms (Transforms): Data transformations to be applied.
        include_polygons (bool): Flag indicating whether to include polygons in the dataset.
            If set to False, polygons will be converted to bitmaps, and bitmaps will be used for training.
        **kwargs: Additional keyword arguments passed to the base class.
    """
    
    def __init__(self, dm_subset: DatasetSubset, transforms: Transforms, include_polygons: bool, **kwargs) -> None:
        super().__init__(dm_subset, transforms, **kwargs)
        self.include_polygons = include_polygons
    
    def _get_item_impl(self, index: int) -> VisualPromptingDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
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
        bboxes = np.array(gt_bboxes, dtype=np.float32)
        bboxes[:, 2:] += bboxes[:, :2]

        masks = np.stack(gt_masks, axis=0) if gt_masks else np.zeros((0, *img_shape), dtype=bool)
        labels = np.array(gt_labels, dtype=np.int64)

        # set entity without masks to avoid resizing masks
        entity = VisualPromptingDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
            ),
            bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            masks=tv_tensors.Mask(masks, dtype=torch.uint8) if isinstance(self.transforms, tvt_v2.Transform) else None,
            labels=torch.as_tensor(labels),
            polygons=gt_polygons,
        )
        transformed_entity = self._apply_transforms(entity)

        # insert masks to transformed_entity
        if not isinstance(self.transforms, tvt_v2.Transform):
            transformed_entity.masks = tv_tensors.Mask(masks, dtype=torch.uint8)
        return transformed_entity

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect VisualPromptingDataEntity into VisualPromptingBatchDataEntity in data loader."""
        return VisualPromptingBatchDataEntity.collate_fn
