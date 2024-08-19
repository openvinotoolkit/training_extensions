# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX detection data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
class KeypointDetDataEntity(OTXDataEntity):
    """Data entity for keypoint detection task.

    :param bboxes: Bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: Bbox labels as integer indices
    :param keypoints: keypoint annotations
        ([[x1, y1], [x2, y2], ...]) format with absolute coordinate values
    :param keypoints_visible: keypoint visibilities with binary values
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.KEYPOINT_DETECTION

    bboxes: tv_tensors.BoundingBoxes
    labels: LongTensor
    keypoints: np.ndarray
    keypoints_visible: np.ndarray
    bbox_info: BboxInfo


@dataclass
class KeypointDetPredEntity(OTXPredEntity, KeypointDetDataEntity):
    """Data entity to represent the keypoint detection model output prediction."""


@dataclass
class KeypointDetBatchDataEntity(OTXBatchDataEntity[KeypointDetDataEntity]):
    """Data entity for keypoint detection task.

    :param bboxes: A list of bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: A list of bbox labels as integer indices
    :param keypoints: keypoint annotations
        ([[x1, y1], [x2, y2], ...]) format with absolute coordinate values
    :param keypoints_visible: keypoint visibilities with binary values
    """

    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[LongTensor]
    bbox_info: list[BboxInfo]
    keypoints: list[np.ndarray]
    keypoints_visible: list[np.ndarray]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.KEYPOINT_DETECTION

    @classmethod
    def collate_fn(
        cls,
        entities: list[KeypointDetDataEntity],
        stack_images: bool = True,
    ) -> KeypointDetBatchDataEntity:
        """Collection function to collect `KeypointDetDataEntity` into `KeypointDetBatchDataEntity` in data loader.

        Args:
            entities: List of `KeypointDetDataEntity`.
            stack_images: If True, return 4D B x C x H x W image tensor.
                Otherwise return a list of 3D C x H x W image tensor.

        Returns:
            Collated `KeypointDetBatchDataEntity`
        """
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        bboxes, labels, bbox_info, kpts, kpts_visible = [], [], [], [], []
        for entity in entities:
            bboxes.append(entity.bboxes)
            labels.append(entity.labels)
            bbox_info.append(entity.bbox_info)
            kpts.append(entity.keypoints)
            kpts_visible.append(entity.keypoints_visible)
        return KeypointDetBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            bboxes=bboxes,
            labels=labels,
            bbox_info=bbox_info,
            keypoints=kpts,
            keypoints_visible=kpts_visible,
        )

    def pin_memory(self) -> KeypointDetBatchDataEntity:
        """Pin memory for member tensor variables."""
        return (
            super()
            .pin_memory()
            .wrap(
                bboxes=[tv_tensors.wrap(bbox.pin_memory(), like=bbox) for bbox in self.bboxes],
                labels=[label.pin_memory() for label in self.labels],
                bbox_info=self.bbox_info,
                keypoints=self.keypoints,
                keypoints_visible=self.keypoints_visible,
            )
        )


@dataclass
class KeypointDetBatchPredEntity(OTXBatchPredEntity, KeypointDetBatchDataEntity):
    """Data entity to represent model output predictions for keypoint detection task."""
