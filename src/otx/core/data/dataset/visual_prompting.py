# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXVisualPromptingDataset."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Callable, Literal

import torch
from datumaro import Bbox as dmBbox
from datumaro import Dataset as dmDataset
from datumaro import Image as dmImage
from datumaro import Mask as dmMask
from datumaro import Points as dmPoints
from datumaro import Polygon as dmPolygon
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import convert_bounding_box_format, to_image
from torchvision.tv_tensors import BoundingBoxes as tvBoundingBoxes
from torchvision.tv_tensors import BoundingBoxFormat as tvBoundingBoxFormat
from torchvision.tv_tensors import Mask as tvMask

from otx.core.data.entity.base import ImageInfo, Points
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingDataEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingDataEntity,
    ZeroShotVisualPromptingLabel,
)
from otx.core.types.label import NullLabelInfo
from otx.core.utils.mask_util import polygon_to_bitmap

from .base import OTXDataset, Transforms


class OTXVisualPromptingDataset(OTXDataset[VisualPromptingDataEntity]):
    """OTXDataset class for visual prompting.

    Args:
        dm_subset (dmDataset): The subset of the dataset.
        transforms (Transforms): Data transformations to be applied.
        use_bbox (bool): Whether to use bounding box prompt.
            If both use_bbox and use_point are False, use_bbox is set to True as default.
            If both are True, divide the probability into both.
            Defaults to True.
        use_point (bool): Whether to use point prompt.
            If both use_bbox and use_point are False, use_bbox is set to True as default.
            If both are True, divide the probability into both.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the base class.
    """

    def __init__(
        self,
        dm_subset: dmDataset,
        transforms: Transforms,
        use_bbox: bool = True,
        use_point: bool = False,
        stack_images: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(dm_subset, transforms, stack_images=stack_images, **kwargs)
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

        self.label_info = NullLabelInfo()

    def _get_item_impl(self, index: int) -> VisualPromptingDataEntity | None:
        item = self.dm_subset[index]
        img = item.media_as(dmImage)
        img_data, img_shape = self._get_img_data_and_shape(img)

        gt_bboxes, gt_points = [], []
        gt_masks = defaultdict(list)
        gt_polygons = defaultdict(list)
        gt_labels = defaultdict(list)

        for annotation in item.annotations:
            if isinstance(annotation, dmPolygon):
                mask = tvMask(polygon_to_bitmap([annotation], *img_shape)[0])
                mask_points = torch.nonzero(mask)
                if len(mask_points) == 0:
                    # skip very small region
                    continue

                if torch.rand(1) < self.prob:
                    # get bbox
                    bbox = tvBoundingBoxes(
                        annotation.get_bbox(),
                        format=tvBoundingBoxFormat.XYWH,
                        canvas_size=img_shape,
                        dtype=torch.float32,
                    )
                    bbox = convert_bounding_box_format(bbox, new_format=tvBoundingBoxFormat.XYXY)
                    gt_bboxes.append(bbox)
                    gt_labels["bboxes"].append(annotation.label)
                    gt_masks["bboxes"].append(mask)
                    gt_polygons["bboxes"].append(annotation)
                else:
                    # get point
                    if item.subset == "train":
                        # get random point from the mask
                        idx_chosen = torch.randperm(len(mask_points[0]))[0]
                        point = Points(
                            (mask_points[1][idx_chosen], mask_points[0][idx_chosen]),
                            canvas_size=img_shape,
                            dtype=torch.float32,
                        )
                    else:
                        # get center point
                        point = Points(
                            torch.tensor(annotation.get_points()).mean(dim=0),
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
        labels = {prompt_type: torch.as_tensor(values, dtype=torch.int64) for prompt_type, values in gt_labels.items()}
        masks = tvMask(
            torch.stack(gt_masks.get("bboxes", []) + gt_masks.get("points", []), dim=0),
            dtype=torch.uint8,
        )
        polygons = gt_polygons.get("bboxes", []) + gt_polygons.get("points", [])

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

        if transformed_entity is None:
            msg = "This is not allowed."
            raise RuntimeError(msg)

        # insert masks to transformed_entity
        return transformed_entity.wrap(masks=masks)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect VisualPromptingDataEntity into VisualPromptingBatchDataEntity in data loader."""  # noqa: E501
        return partial(VisualPromptingBatchDataEntity.collate_fn, stack_images=self.stack_images)


class OTXZeroShotVisualPromptingDataset(OTXDataset[ZeroShotVisualPromptingDataEntity]):
    """OTXDataset class for zero-shot visual prompting.

    Args:
        dm_subset (dmDataset): The subset of the dataset.
        transforms (Transforms): Data transformations to be applied.
        use_bbox (bool): Whether to use bounding box prompt.
            If both use_bbox and use_point are False, use_bbox is set to True as default.
            If both are True, divide the probability into both.
            Defaults to True.
        use_point (bool): Whether to use point prompt.
            If both use_bbox and use_point are False, use_bbox is set to True as default.
            If both are True, divide the probability into both.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the base class.
    """

    def __init__(
        self,
        dm_subset: dmDataset,
        transforms: Transforms,
        use_bbox: bool = True,
        use_point: bool = False,
        stack_images: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(dm_subset, transforms, stack_images=stack_images, **kwargs)
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

        self.label_info = NullLabelInfo()

    def _get_item_impl(self, index: int) -> ZeroShotVisualPromptingDataEntity | None:
        item = self.dm_subset[index]
        img = item.media_as(dmImage)
        img_data, img_shape = self._get_img_data_and_shape(img)

        gt_prompts: list[tvBoundingBoxes | Points] = []
        gt_masks: list[tvMask] = []
        gt_polygons: list[dmPolygon] = []
        gt_labels: dict[Literal["prompts", "polygons", "masks"], list[int]] = defaultdict(list)
        for annotation in item.annotations:
            if isinstance(annotation, dmPolygon):
                # generate prompts from polygon
                mask = tvMask(polygon_to_bitmap([annotation], *img_shape)[0])
                mask_points = torch.nonzero(mask)
                if len(mask_points) == 0:
                    # skip very small region
                    continue

                if torch.rand(1) < self.prob:
                    # get bbox
                    bbox = tvBoundingBoxes(
                        annotation.get_bbox(),
                        format=tvBoundingBoxFormat.XYWH,
                        canvas_size=img_shape,
                        dtype=torch.float32,
                    )
                    bbox = convert_bounding_box_format(bbox, new_format=tvBoundingBoxFormat.XYXY)
                    gt_prompts.append(bbox)
                else:
                    # get center point
                    point = Points(
                        torch.tensor(annotation.get_points()).mean(dim=0),
                        canvas_size=img_shape,
                        dtype=torch.float32,
                    )
                    gt_prompts.append(point)

                gt_labels["prompts"].append(annotation.label)
                gt_labels["polygons"].append(annotation.label)
                gt_labels["masks"].append(annotation.label)
                gt_masks.append(mask)
                gt_polygons.append(annotation)

            # TODO(sungchul): for mask, bounding box, and point annotation
            elif isinstance(annotation, (dmBbox, dmMask, dmPoints)):
                pass

        if not gt_prompts:
            return None

        labels = {
            str(prompt_type): torch.as_tensor(values, dtype=torch.int64) for prompt_type, values in gt_labels.items()
        }
        masks = tvMask(torch.stack(gt_masks, dim=0), dtype=torch.uint8)

        return ZeroShotVisualPromptingDataEntity(
            image=to_image(img_data),
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
            ),
            masks=masks,
            labels=ZeroShotVisualPromptingLabel(**labels),
            polygons=gt_polygons,
            prompts=gt_prompts,
        )

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect ZeroShotVisualPromptingDataEntity into ZeroShotVisualPromptingBatchDataEntity in data loader."""  # noqa: E501
        return partial(ZeroShotVisualPromptingBatchDataEntity.collate_fn, stack_images=self.stack_images)
