# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX detection data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch

from torchvision import tv_tensors

from otx.core.data.entity.base import (
    BboxInfo,
    OTXBatchDataEntity,
    OTXBatchPredEntity,
    OTXDataEntity,
    OTXPredEntity,
)
from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    import numpy as np
    from torch import LongTensor


@register_pytree_node
@dataclass
class Det3DDataEntity(OTXDataEntity):
    """Data entity for detection task.

    :param bboxes: Bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: Bbox labels as integer indices
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.OBJECT_DETECTION_3D

    bboxes_2d: tv_tensors.BoundingBoxes
    calib_p2: np.ndarray
    calibs: np.ndarray
    bboxes_3d: np.ndarray
    size_2d: np.ndarray
    size_3d: np.ndarray
    src_size_3d: np.ndarray
    depth: np.ndarray
    heading_bin: np.ndarray
    heading_res: np.ndarray
    mask_2d: np.ndarray
    indices: np.ndarray
    labels: LongTensor

@dataclass
class Det3DPredEntity(OTXPredEntity, Det3DDataEntity):
    """Data entity to represent the detection model output prediction."""


@dataclass
class Det3DBatchDataEntity(OTXBatchDataEntity[Det3DDataEntity]):
    """Data entity for detection task.

    :param bboxes: A list of bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: A list of bbox labels as integer indices
    """ # TODO(Kirill): UPDATE!

    bboxes_2d: list[tv_tensors.BoundingBoxes]
    calib_p2: list[np.ndarray]
    calibs: list[np.ndarray]
    bboxes_3d: list[np.ndarray]
    size_2d: list[np.ndarray]
    size_3d: list[np.ndarray]
    src_size_3d: list[np.ndarray]
    depth: list[np.ndarray]
    heading_bin: list[np.ndarray]
    heading_res: list[np.ndarray]
    mask_2d: list[np.ndarray]
    indices: list[np.ndarray]
    labels: list[LongTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.OBJECT_DETECTION_3D

    @classmethod
    def collate_fn(
        cls,
        entities: list[Det3DDataEntity],
        stack_images: bool = True,
    ) -> Det3DBatchDataEntity:
        """Collection function to collect `DetDataEntity` into `DetBatchDataEntity` in data loader.

        Args:
            entities: List of `DetDataEntity`.
            stack_images: If True, return 4D B x C x H x W image tensor.
                Otherwise return a list of 3D C x H x W image tensor.

        Returns:
            Collated `DetBatchDataEntity`
        """
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        batch_input_shape = tuple(batch_data.images[0].size()[-2:])
        for info in batch_data.imgs_info:
            info.batch_input_shape = batch_input_shape
        return Det3DBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            bboxes_2d=[entity.bboxes_2d for entity in entities],
            labels=[entity.labels for entity in entities],
            calib_p2=[entity.calib_p2 for entity in entities],
            calibs=[entity.calibs for entity in entities],
            bboxes_3d=[entity.bboxes_3d for entity in entities],
            size_2d=[entity.size_2d for entity in entities],
            size_3d=[entity.size_3d for entity in entities],
            src_size_3d=[entity.src_size_3d for entity in entities],
            depth=[entity.depth for entity in entities],
            heading_bin=[entity.heading_bin for entity in entities],
            heading_res=[entity.heading_res for entity in entities],
            mask_2d=[entity.mask_2d for entity in entities],
            indices=[entity.indices for entity in entities],
        )

    def pin_memory(self) -> Det3DBatchDataEntity:
        """Pin memory for member tensor variables."""
        return (
            super()
            .pin_memory()
            .wrap(
                bboxes_2d=[tv_tensors.wrap(bbox.pin_memory(), like=bbox) for bbox in self.bboxes_2d],
                labels=[label.pin_memory() for label in self.labels],
                calibs=self.calibs,
                calib_p2=self.calib_p2,
                bboxes_3d=self.bboxes_3d,
                size_2d=self.size_2d,
                size_3d=self.size_3d,
                src_size_3d=self.src_size_3d,
                depth=self.depth,
                heading_bin=self.heading_bin,
                heading_res=self.heading_res,
                mask_2d=self.mask_2d,
                indices=self.indices,
            )
        )


@dataclass
class Det3DBatchPredEntity(OTXBatchPredEntity, Det3DBatchDataEntity):
    """Data entity to represent model output predictions for detection task."""
