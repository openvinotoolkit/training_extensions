# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX diffusion data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from otx.core.data.entity.base import (
    OTXBatchDataEntity,
    OTXBatchPredEntity,
    OTXDataEntity,
    OTXPredEntity,
)
from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from torch import Tensor


@register_pytree_node
@dataclass
class DiffusionDataEntity(OTXDataEntity):
    """Data entity for diffusion task.

    :param input_ids: caption corresponding to the image
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.DIFFUSION

    caption: str


@dataclass
class DiffusionPredEntity(OTXPredEntity, DiffusionDataEntity):
    """Data entity to represent the keypoint detection model output prediction."""


@dataclass
class DiffusionBatchDataEntity(OTXBatchDataEntity[DiffusionDataEntity]):
    """Batch data entity for diffusion task.

    :param images: a list of original images
    :param noise: a list of generated Gaussian noises corresponding to the images
    """

    images: Tensor

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.DIFFUSION

    captions: list[str]

    @classmethod
    def collate_fn(
        cls,
        entities: list[DiffusionDataEntity],
        stack_images: bool = True,
    ) -> DiffusionBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader.

        Args:
            entities: List of OTX data entities.
            stack_images: If True, return 4D B x C x H x W image tensor.
                Otherwise return a list of 3D C x H x W image tensor.

        Returns:
            Collated OTX batch data entity
        """
        batch_data = super().collate_fn(entities, stack_images)
        return DiffusionBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            captions=[entity.caption for entity in entities],
        )


@dataclass
class DiffusionBatchPredEntity(OTXBatchPredEntity):
    """Data entity to represent model output predictions for keypoint detection task."""

    images: Tensor
