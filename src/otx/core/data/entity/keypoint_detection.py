# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX detection data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchvision import tv_tensors

from otx.core.data.entity.base import (
    OTXBatchDataEntity,
    OTXBatchPredEntity,
    OTXDataEntity,
    OTXPredEntity,
)
from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from torch import BoolTensor, LongTensor


@register_pytree_node
@dataclass
class KeypointDetDataEntity(OTXDataEntity):
    """Data entity for detection task.

    :param bboxes: Bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: Bbox labels as integer indices
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.KEYPOINT_DETECTION

    bboxes: tv_tensors.BoundingBoxes
    labels: LongTensor
    keypoints: tv_tensors.TVTensor
    keypoints_visible: BoolTensor
    keypoint_x_labels: tv_tensors.TVTensor
    keypoint_y_labels: tv_tensors.TVTensor
    keypoint_weights: tv_tensors.TVTensor


@dataclass
class KeypointDetPredEntity(OTXPredEntity, KeypointDetDataEntity):
    """Data entity to represent the detection model output prediction."""


@dataclass
class KeypointDetBatchDataEntity(OTXBatchDataEntity[KeypointDetDataEntity]):
    """Data entity for detection task.

    :param bboxes: A list of bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: A list of bbox labels as integer indices
    """

    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[LongTensor]
    keypoints: list[tv_tensors.TVTensor]
    keypoints_visible: list[BoolTensor]
    keypoint_x_labels: list[tv_tensors.TVTensor]
    keypoint_y_labels: list[tv_tensors.TVTensor]
    keypoint_weights: list[tv_tensors.TVTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.DETECTION

    @classmethod
    def collate_fn(
        cls,
        entities: list[KeypointDetDataEntity],
        stack_images: bool = True,
    ) -> KeypointDetBatchDataEntity:
        """Collection function to collect `DetDataEntity` into `DetBatchDataEntity` in data loader.

        Args:
            entities: List of `DetDataEntity`.
            stack_images: If True, return 4D B x C x H x W image tensor.
                Otherwise return a list of 3D C x H x W image tensor.

        Returns:
            Collated `DetBatchDataEntity`
        """
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        # batch_input_shape = tuple(batch_data.images[0].size()[-2:])
        # for info in batch_data.imgs_info:
        #     info.batch_input_shape = batch_input_shape
        return KeypointDetBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            bboxes=[entity.bboxes for entity in entities],
            keypoints=[entity.keypoints for entity in entities],
            keypoints_visible=[entity.keypoints_visible for entity in entities],
            keypoint_x_labels=[entity.keypoint_x_labels for entity in entities],
            keypoint_y_labels=[entity.keypoint_y_labels for entity in entities],
            keypoint_weights=[entity.keypoint_weights for entity in entities],
            labels=[entity.labels for entity in entities],
        )

    def pin_memory(self) -> KeypointDetBatchDataEntity:
        """Pin memory for member tensor variables."""
        return (
            super()
            .pin_memory()
            .wrap(
                bboxes=[tv_tensors.wrap(bbox.pin_memory(), like=bbox) for bbox in self.bboxes],
                keypoints=[tv_tensors.wrap(keypoints.pin_memory(), like=keypoints) for keypoints in self.keypoints],
                keypoints_visible=[
                    tv_tensors.wrap(visible.pin_memory(), like=visible) for visible in self.keypoints_visible
                ],
                keypoint_x_labels=[
                    tv_tensors.wrap(keypoint_x_labels.pin_memory(), like=keypoint_x_labels)
                    for keypoint_x_labels in self.keypoint_x_labels
                ],
                keypoint_y_labels=[
                    tv_tensors.wrap(keypoint_y_labels.pin_memory(), like=keypoint_y_labels)
                    for keypoint_y_labels in self.keypoint_y_labels
                ],
                keypoint_weights=[
                    tv_tensors.wrap(keypoint_weights.pin_memory(), like=keypoint_weights)
                    for keypoint_weights in self.keypoint_weights
                ],
                labels=[label.pin_memory() for label in self.labels],
            )
        )


@dataclass
class KeypointDetBatchPredEntity(OTXBatchPredEntity, KeypointDetBatchDataEntity):
    """Data entity to represent model output predictions for detection task."""
