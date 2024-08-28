# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTXKeypointDetectionDataset."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Callable, List, Union

import numpy as np
import torch
from datumaro import AnnotationType, Bbox, Dataset, DatasetSubset, Image, Points
from torchvision import tv_tensors

from otx.core.data.entity.base import BboxInfo, ImageInfo
from otx.core.data.entity.keypoint_detection import KeypointDetBatchDataEntity, KeypointDetDataEntity
from otx.core.data.mem_cache import NULL_MEM_CACHE_HANDLER, MemCacheHandlerBase
from otx.core.data.transform_libs.torchvision import Compose
from otx.core.types.image import ImageColorChannel
from otx.core.types.label import LabelInfo, NullLabelInfo

from .base import OTXDataset

Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]


class OTXKeypointDetectionDataset(OTXDataset[KeypointDetDataEntity]):
    """OTXDataset class for keypoint detection task."""

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = NULL_MEM_CACHE_HANDLER,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
    ) -> None:
        super().__init__(
            dm_subset,
            transforms,
            mem_cache_handler,
            mem_cache_img_max_size,
            max_refetch,
            image_color_channel,
            stack_images,
            to_tv_image,
        )

        self.dm_subset = self._get_single_bbox_dataset(dm_subset)

        if self.dm_subset.categories():
            self.label_info = LabelInfo(
                label_names=self.dm_subset.categories()[AnnotationType.points][0].labels,
                label_groups=[],
            )
        else:
            self.label_info = NullLabelInfo()

    def _get_single_bbox_dataset(self, dm_subset: DatasetSubset) -> Dataset:
        """Method for splitting dataset items into multiple items for each bbox/keypoint."""
        dm_items = []
        for item in dm_subset:
            new_items = defaultdict(list)
            for ann in item.annotations:
                if isinstance(ann, (Bbox, Points)):
                    new_items[ann.id].append(ann)
            for ann_id, anns in new_items.items():
                available_types = []
                for ann in anns:
                    if isinstance(ann, Bbox) and (ann.w <= 0 or ann.h <= 0):
                        continue
                    if isinstance(ann, Points) and max(ann.points) <= 0:
                        continue
                    available_types.append(ann.type)
                if available_types != [AnnotationType.points, AnnotationType.bbox]:
                    continue
                dm_items.append(item.wrap(id=item.id + "_" + str(ann_id), annotations=anns))
        return Dataset.from_iterable(dm_items, categories=self.dm_subset.categories())

    def _get_item_impl(self, index: int) -> KeypointDetDataEntity | None:
        item = self.dm_subset[index]
        img = item.media_as(Image)
        ignored_labels: list[int] = []  # This should be assigned form item
        img_data, img_shape = self._get_img_data_and_shape(img)

        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]
        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        keypoint_anns = [ann for ann in item.annotations if isinstance(ann, Points)]
        keypoints = (
            np.stack([ann.points for ann in keypoint_anns], axis=0).astype(np.float32)
            if len(keypoint_anns) > 0
            else np.zeros((0, len(self.label_info.label_names) * 2), dtype=np.float32)
        ).reshape(-1, 2)
        keypoints_visible = np.minimum(1, keypoints)[..., 0]

        bbox_center = np.array(img_shape) / 2.0
        bbox_scale = np.array(img_shape)
        bbox_rotation = 0.0

        entity = KeypointDetDataEntity(
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
            labels=torch.as_tensor([ann.label for ann in bbox_anns]),
            bbox_info=BboxInfo(center=bbox_center, scale=bbox_scale, rotation=bbox_rotation),
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect KeypointDetDataEntity into KeypointDetBatchDataEntity in data loader."""
        return partial(KeypointDetBatchDataEntity.collate_fn, stack_images=self.stack_images)
