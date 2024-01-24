# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXVisualPromptingDataset."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torchvision.transforms.v2.functional as F  # noqa: N812
from datumaro import DatasetSubset, Image, Polygon
from torchvision import tv_tensors
from collections import defaultdict

from otx.core.data.entity.base import ImageInfo, Points
from otx.core.data.entity.visual_prompting import VisualPromptingBatchDataEntity, VisualPromptingDataEntity
from otx.core.utils.mask_util import polygon_to_bitmap

from .base import OTXDataset, Transforms


class OTXVisualPromptingDataset(OTXDataset[VisualPromptingDataEntity]):
    """OTXDataset class for visual prompting.

    Args:
        dm_subset (DatasetSubset): The subset of the dataset.
        transforms (Transforms): Data transformations to be applied.
        **kwargs: Additional keyword arguments passed to the base class.
    """

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        use_bbox: bool = True,
        use_point: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(dm_subset, transforms, **kwargs)
        if not use_bbox and not use_point:
            # if both are False, use bbox as default
            use_bbox = True
        self.prob = 1.0  # if using only bbox prompt
        if use_bbox and use_point:
            # if using both prompts, divide prob into both
            self.prob = 0.5
        if not use_bbox and use_point:
            # if using only point prompt
            self.prob = 0.0

    def _get_item_impl(self, index: int) -> VisualPromptingDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        gt_bboxes, gt_points = [], []
        gt_masks = defaultdict(list)
        gt_polygons = defaultdict(list)
        gt_labels = defaultdict(list)

        for annotation in item.annotations:
            if isinstance(annotation, Polygon):
                mask = polygon_to_bitmap([annotation], *img_shape)[0]
                mask_points = np.nonzero(mask)
                if len(mask_points[0]) == 0:
                    # skip very small region
                    continue

                if np.random.rand() < self.prob:  # noqa: NPY002
                    # get bbox
                    bbox = tv_tensors.BoundingBoxes(
                        annotation.get_bbox(),
                        format=tv_tensors.BoundingBoxFormat.XYWH,
                        canvas_size=img_shape,
                        dtype=torch.float32,
                    )
                    bbox = F._meta.convert_bounding_box_format(bbox, new_format=tv_tensors.BoundingBoxFormat.XYXY)  # noqa: SLF001
                    gt_bboxes.append(bbox)
                    gt_labels["bboxes"].append(annotation.label)
                    gt_masks["bboxes"].append(mask)
                    gt_polygons["bboxes"].append(annotation)
                else:
                    # get point
                    if self.dm_subset.name == "train":
                        # get random point from the mask
                        idx_chosen = np.random.permutation(len(mask_points[0]))[0]  # noqa: NPY002
                        point = Points(
                            (mask_points[1][idx_chosen], mask_points[0][idx_chosen]),
                            canvas_size=img_shape,
                            dtype=torch.float32,
                        )
                    else:
                        # get center point
                        point = Points(
                            np.array(annotation.get_points()).mean(axis=0),
                            canvas_size=img_shape,
                            dtype=torch.float32,
                        )
                    gt_points.append(point)
                    gt_labels["points"].append(annotation.label)
                    gt_masks["points"].append(mask)
                    gt_polygons["points"].append(annotation)

        assert (  # noqa: S101
            len(gt_bboxes) > 0 or len(gt_points) > 0
        ), "At least one of both #bounding box and #point prompts must be greater than 0."

        bboxes = tv_tensors.wrap(torch.cat(gt_bboxes, dim=0), like=gt_bboxes[0]) if len(gt_bboxes) > 0 else None
        points = tv_tensors.wrap(torch.stack(gt_points, dim=0), like=gt_points[0]) if len(gt_points) > 0 else None
        labels = torch.as_tensor(gt_labels.get("points", []) + gt_labels.get("bboxes", []), dtype=torch.int64)
        masks = tv_tensors.Mask(
            np.stack(gt_masks.get("points", []) + gt_masks.get("bboxes", []), axis=0),
            dtype=torch.uint8,
        )
        polygons = gt_polygons.get("points", []) + gt_polygons.get("bboxes", [])

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
            polygons=polygons,
            points=points,
            bboxes=bboxes,
        )
        transformed_entity = self._apply_transforms(entity)

        # insert masks to transformed_entity
        transformed_entity.masks = masks  # type: ignore[union-attr]
        return transformed_entity

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect VisualPromptingDataEntity into VisualPromptingBatchDataEntity in data loader."""  # noqa: E501
        return VisualPromptingBatchDataEntity.collate_fn
