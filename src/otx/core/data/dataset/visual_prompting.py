# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXVisualPromptingDataset."""

from __future__ import annotations

from typing import Callable

import numpy as np
from collections import defaultdict
import torch
from datumaro import DatasetSubset, Image, Polygon
from torchvision import tv_tensors
import torchvision.transforms.v2.functional as F

from otx.core.data.entity.base import ImageInfo, Points
from otx.core.data.entity.visual_prompting import VisualPromptingBatchDataEntity, VisualPromptingDataEntity
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

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        include_polygons: bool,
        use_bbox: bool = True,
        use_point: bool = False,
        **kwargs
    ) -> None:
        super().__init__(dm_subset, transforms, **kwargs)
        self.include_polygons = include_polygons
        if not use_bbox and not use_point:
            # if both are False, use bbox as default
            use_bbox = True
        self.prob = 1. # if using only bbox prompt
        if use_bbox and use_point:
            # if using both prompts, divide prob into both
            self.prob = 0.5
        if not use_bbox and use_point:
            # if using only point prompt
            self.prob = 0.

    def _get_item_impl(self, index: int) -> VisualPromptingDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        gt_bboxes, gt_points, gt_masks, gt_polygons = [], [], [], []
        gt_labels = defaultdict(list)

        for annotation in item.annotations:
            if isinstance(annotation, Polygon):
                mask = polygon_to_bitmap([annotation], *img_shape)[0]
                mask_points = np.nonzero(mask)
                if len(mask_points[0]) == 0:
                    # skip very small region
                    continue

                if self.include_polygons:
                    gt_polygons.append(annotation)
                else:
                    gt_masks.append(mask)

                if np.random.rand() < self.prob:
                    # get bbox
                    bbox = tv_tensors.BoundingBoxes(
                        annotation.get_bbox(),
                        format=tv_tensors.BoundingBoxFormat.XYWH,
                        canvas_size=img_shape,
                        dtype=torch.float32)
                    bbox = F._meta.convert_bounding_box_format(
                        bbox,
                        new_format=tv_tensors.BoundingBoxFormat.XYXY)
                    gt_bboxes.append(bbox)
                    gt_labels["bboxes"].append(annotation.label)
                else:
                    # get point
                    if self.dm_subset.name == "train":
                        # get random point from the mask
                        idx_chosen = np.random.permutation(len(mask_points[0]))[0]
                        point = Points(
                            (mask_points[1][idx_chosen], mask_points[0][idx_chosen]),
                            canvas_size=img_shape,
                            dtype=torch.float32)
                    else:
                        # get center point
                        point = Points(
                            np.array(annotation.get_points()).mean(axis=0),
                            canvas_size=img_shape,
                            dtype=torch.float32)
                    gt_points.append(point)
                    gt_labels["points"].append(annotation.label)
        
        assert len(gt_bboxes) > 0 or len(gt_points) > 0, \
            "At least one of both #bounding box and #point prompts must be greater than 0."

        bboxes = tv_tensors.wrap(torch.cat(gt_bboxes, dim=0), like=gt_bboxes[0]) if len(gt_bboxes) > 0 else None
        points = tv_tensors.wrap(torch.stack(gt_points, dim=0), like=gt_points[0]) if len(gt_points) > 0 else None
        labels = torch.as_tensor(gt_labels.get("points", []) + gt_labels.get("bboxes", []), dtype=torch.int64)
        
        # set entity without masks to avoid resizing masks
        entity = VisualPromptingDataEntity(
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
            ),
            masks=None,
            labels=labels,
            polygons=gt_polygons,
            points=points,
            bboxes=bboxes,
        )
        transformed_entity = self._apply_transforms(entity)

        # insert masks to transformed_entity
        masks = np.stack(gt_masks, axis=0) if gt_masks else np.zeros((0, *img_shape), dtype=bool)
        transformed_entity.masks = tv_tensors.Mask(masks, dtype=torch.uint8)  # type: ignore[union-attr]
        return transformed_entity

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect VisualPromptingDataEntity into VisualPromptingBatchDataEntity in data loader."""  # noqa: E501
        return VisualPromptingBatchDataEntity.collate_fn
