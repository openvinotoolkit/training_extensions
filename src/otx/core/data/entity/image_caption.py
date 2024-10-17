# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX IMAGE_CAPTIONING data entities."""
from __future__ import annotations

from dataclasses import dataclass

from otx.core.data.entity.base import (
    OTXBatchDataEntity,
    OTXBatchPredEntity,
    OTXDataEntity,
    OTXPredEntity,
)
from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType


@register_pytree_node
@dataclass
class ImageCaptionDataEntity(OTXDataEntity):
    """Data entity for LANGUAGE-IMAGE task."""

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.IMAGE_CAPTIONING

    captions: list[list[str]] | list[str]


@dataclass
class ImageCaptionPredEntity(OTXPredEntity, ImageCaptionDataEntity):
    """Data entity to represent the LANGUAGE-IMAGE model output prediction."""

@dataclass
class ImageCaptionBatchDataEntity(OTXBatchDataEntity[ImageCaptionDataEntity]):
    """Data entity for LANGUAGE-IMAGE task."""

    captions: list[list[str]] | list[str]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.IMAGE_CAPTIONING

    @classmethod
    def collate_fn(
        cls,
        entities: list[ImageCaptionDataEntity],
        stack_images: bool = True,
    ) -> ImageCaptionBatchDataEntity:
        """Collate function for ImageTextBatchDataEntity."""
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        captions = [data.captions for data in batch_data]
        return cls(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            captions=captions,
        )

@dataclass
class ImageCaptionBatchPredEntity(OTXBatchPredEntity, ImageCaptionBatchDataEntity):
    """Data entity to represent the LANGUAGE-IMAGE model output prediction."""
