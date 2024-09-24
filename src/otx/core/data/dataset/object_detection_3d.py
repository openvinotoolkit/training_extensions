# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX3DObjectDetectionDataset."""

from __future__ import annotations

from functools import partial
from typing import Callable, List, Union

import numpy as np
import torch
from otx.core.data.dataset.kitti_3d.kitti3d import KITTI_Dataset
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.object_detection_3d import Det3DBatchDataEntity, Det3DDataEntity
from otx.core.data.mem_cache import NULL_MEM_CACHE_HANDLER, MemCacheHandlerBase
from otx.core.data.transform_libs.torchvision import Compose
from otx.core.types.image import ImageColorChannel
from otx.core.types.label import LabelInfo
from torchvision import tv_tensors

from .base import OTXDataset

Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]


class OTX3DObjectDetectionDataset(OTXDataset[Det3DDataEntity]):
    """OTXDataset class for detection task."""

    def __init__(
        self,
        dm_subset: KITTI_Dataset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = NULL_MEM_CACHE_HANDLER,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
    ) -> None:
        self.dm_subset = dm_subset
        self.transforms = transforms
        self.mem_cache_handler = mem_cache_handler
        self.mem_cache_img_max_size = mem_cache_img_max_size
        self.max_refetch = max_refetch
        self.image_color_channel = image_color_channel
        self.stack_images = stack_images
        self.to_tv_image = to_tv_image
        self.label_info = LabelInfo(label_names=self.dm_subset.class_name, label_groups=[self.dm_subset.class_name])

    def _get_item_impl(self, index: int) -> Det3DDataEntity | None:
        inputs, calib_entity, targets, info, objects = self.dm_subset[index]
        entity = Det3DDataEntity(
            image=torch.tensor(inputs),
            img_info=ImageInfo(
                img_idx=index,
                img_shape=(384, 1280),
                ori_shape=info["img_size"],
                image_color_channel=self.image_color_channel,
                ignored_labels=[],
            ),
            boxes=tv_tensors.BoundingBoxes(
                targets["boxes"],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(384, 1280),
                dtype=torch.float32,
            ),
            labels=torch.as_tensor(targets["labels"], dtype=torch.long),
            calib_matrix=torch.as_tensor(calib_entity.P2, dtype=torch.float32),
            boxes_3d=torch.as_tensor(targets["boxes_3d"], dtype=torch.float32),
            size_2d=torch.as_tensor(targets["size_2d"], dtype=torch.float32),
            size_3d=torch.as_tensor(targets["size_3d"], dtype=torch.float32),
            depth=torch.as_tensor(targets["depth"], dtype=torch.float32),
            heading_angle=torch.as_tensor(
                np.concatenate([targets["heading_bin"], targets["heading_res"]], axis=1),
                dtype=torch.float32,
            ),
            kitti_label_object=objects,
        )

        return entity

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect DetDataEntity into DetBatchDataEntity in data loader."""
        return partial(Det3DBatchDataEntity.collate_fn, stack_images=self.stack_images)
