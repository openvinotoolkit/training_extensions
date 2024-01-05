# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX tile data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchvision import tv_tensors

from otx.core.types.task import OTXTaskType

from .base import ImageInfo
from .detection import DetDataEntity

if TYPE_CHECKING:
    from torch import LongTensor


@dataclass
class TileDetDataEntity:
    """Data entity for tile task.

    :param entity: A list of OTXDataEntity
    """

    num_tiles: int
    entity_list: list[DetDataEntity]
    ori_bboxes: tv_tensors.BoundingBoxes
    ori_labels: LongTensor

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.DETECTION


@dataclass
class TileBatchDetDataEntity:
    """Batch data entity for tile task."""

    batch_size: int
    batch_tiles: list[list[tv_tensors.Image]]
    batch_tile_infos: list[list[ImageInfo]]
    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[list[LongTensor]]

    @classmethod
    def collate_fn(cls, batch_entities: list[TileDetDataEntity]) -> TileBatchDetDataEntity:
        """Collate function to collect TileDetDataEntity into TileBatchDetDataEntity in data loader."""
        if (batch_size := len(batch_entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        task = batch_entities[0].task

        for tile_entity in batch_entities:
            for entity in tile_entity.entity_list:
                if entity.task != task:
                    msg = "collate_fn() input should include a single OTX task"
                    raise RuntimeError(msg)

                if not isinstance(entity, DetDataEntity):
                    msg = "All entities should be DetDataEntity before collate_fn()"
                    raise RuntimeError(msg)

        return TileBatchDetDataEntity(
            batch_size=batch_size,
            batch_tiles=[[entity.image for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_tile_infos=[
                [entity.img_info for entity in tile_entity.entity_list] for tile_entity in batch_entities
            ],
            bboxes=[tile_entity.ori_bboxes for tile_entity in batch_entities],
            labels=[tile_entity.ori_labels for tile_entity in batch_entities],
        )
