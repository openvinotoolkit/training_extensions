# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX3DObjectDetectionDataset."""

# mypy: ignore-errors

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, List, Union

import numpy as np
from datumaro import Image

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.object_detection_3d import Det3DBatchDataEntity, Det3DDataEntity
from otx.core.data.mem_cache import NULL_MEM_CACHE_HANDLER, MemCacheHandlerBase
from otx.core.data.transform_libs.torchvision import Compose
from otx.core.types.image import ImageColorChannel

from .base import OTXDataset

if TYPE_CHECKING:
    from datumaro import DatasetSubset


Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]


class OTX3DObjectDetectionDataset(OTXDataset[Det3DDataEntity]):
    """OTXDataset class for detection task."""

    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Transforms,
        mem_cache_handler: MemCacheHandlerBase = NULL_MEM_CACHE_HANDLER,
        mem_cache_img_max_size: tuple[int, int] | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = False,
        max_objects: int = 50,
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
        self.max_objects = max_objects
        self.subset_type = list(self.dm_subset.get_subset_info())[-1].split(":")[0]

    def _get_item_impl(self, index: int) -> Det3DDataEntity | None:
        entity = self.dm_subset[index]
        image = entity.media_as(Image)
        image, ori_img_shape = self._get_img_data_and_shape(image)
        calib = self.get_calib_from_file(entity.attributes["calib_path"])
        annotations_copy = deepcopy(entity.annotations)
        original_kitti_format = [obj.attributes for obj in annotations_copy]

        # decode original kitti format for metric calculation
        for i, anno_dict in enumerate(original_kitti_format):
            anno_dict["name"] = (
                self.label_info.label_names[annotations_copy[i].label]
                if self.subset_type != "train"
                else annotations_copy[i].label
            )
            anno_dict["bbox"] = annotations_copy[i].points
            dimension = anno_dict["dimensions"]
            anno_dict["dimensions"] = [dimension[2], dimension[0], dimension[1]]
        original_kitti_format = self._reformate_for_kitti_metric(original_kitti_format)

        entity = Det3DDataEntity(
            image=image,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=ori_img_shape,
                ori_shape=ori_img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=[],
            ),
            boxes=np.zeros((self.max_objects, 4), dtype=np.float32),
            labels=np.zeros((self.max_objects), dtype=np.int8),
            calib_matrix=calib,
            boxes_3d=np.zeros((self.max_objects, 6), dtype=np.float32),
            size_2d=np.zeros((self.max_objects, 2), dtype=np.float32),
            size_3d=np.zeros((self.max_objects, 3), dtype=np.float32),
            depth=np.zeros((self.max_objects, 1), dtype=np.float32),
            heading_angle=np.zeros((self.max_objects, 2), dtype=np.float32),
            original_kitti_format=original_kitti_format,
        )

        return self._apply_transforms(entity)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect DetDataEntity into DetBatchDataEntity in data loader."""
        return partial(Det3DBatchDataEntity.collate_fn, stack_images=self.stack_images)

    def _reformate_for_kitti_metric(self, annotations: dict[str, Any]) -> dict[str, np.array]:
        """Reformat the annotation for KITTI metric."""
        return {
            "name": np.array([obj["name"] for obj in annotations]),
            "alpha": np.array([obj["alpha"] for obj in annotations]),
            "bbox": np.array([obj["bbox"] for obj in annotations]).reshape(-1, 4),
            "dimensions": np.array([obj["dimensions"] for obj in annotations]).reshape(-1, 3),
            "location": np.array([obj["location"] for obj in annotations]).reshape(-1, 3),
            "rotation_y": np.array([obj["rotation_y"] for obj in annotations]),
            "occluded": np.array([obj["occluded"] for obj in annotations]),
            "truncated": np.array([obj["truncated"] for obj in annotations]),
        }

    @staticmethod
    def get_calib_from_file(calib_file: str) -> np.ndarray:
        """Get calibration matrix from txt file (KITTI format)."""
        with open(calib_file) as f:  # noqa: PTH123
            lines = f.readlines()

        obj = lines[2].strip().split(" ")[1:]

        return np.array(obj, dtype=np.float32).reshape(3, 4)
