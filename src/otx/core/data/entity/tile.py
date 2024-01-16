# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX tile data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from otx.core.types.task import OTXTaskType

from .base import ImageInfo, T_OTXBatchDataEntity
from .detection import DetBatchDataEntity, DetDataEntity
from .instance_segmentation import InstanceSegBatchDataEntity, InstanceSegDataEntity

if TYPE_CHECKING:
    from datumaro import Polygon
    from torch import LongTensor
    from torchvision import tv_tensors


T_OTXTileBatchDataEntity = TypeVar(
    "T_OTXTileBatchDataEntity",
    bound="OTXTileBatchDataEntity",
)


@dataclass
class TileDetDataEntity:
    """Data entity for tile task.

    :param entity: A list of OTXDataEntity
    """

    num_tiles: int
    entity_list: list[DetDataEntity]
    tile_attr_list: list[dict[str, int | str]]
    ori_bboxes: tv_tensors.BoundingBoxes
    ori_labels: LongTensor

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.DETECTION


@dataclass
class OTXTileBatchDataEntity(Generic[T_OTXBatchDataEntity]):
    """Batch data entity for tile task.

    Attributes:
        batch_size (int): The size of the batch.
        batch_tiles (list[list[tv_tensors.Image]]): The batch of tile images.
        batch_tile_infos (list[list[ImageInfo]]): The image information about the tiles.
        batch_tile_attr_list (list[list[dict[str, int | str]]]): The tile attributes.
    """

    batch_size: int
    batch_tiles: list[list[tv_tensors.Image]]
    batch_img_infos: list[list[ImageInfo]]
    batch_tile_attr_list: list[list[dict[str, int | str]]]

    def unbind(self) -> list[T_OTXBatchDataEntity]:
        """Unbind batch data entity."""
        raise NotImplementedError


@dataclass
class TileBatchDetDataEntity(OTXTileBatchDataEntity):
    """Batch data entity for tile task."""

    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[LongTensor]

    def unbind(self) -> list[tuple[list[dict[str, int | str]], DetBatchDataEntity]]:
        """Unbind batch data entity for detection task."""
        tiles = [tile for tiles in self.batch_tiles for tile in tiles]
        img_infos = [img_info for img_infos in self.batch_img_infos for img_info in img_infos]
        tile_attr_list = [tile_attr for tile_attrs in self.batch_tile_attr_list for tile_attr in tile_attrs]

        batch_tile_attr_list = [
            tile_attr_list[i : i + self.batch_size] for i in range(0, len(tile_attr_list), self.batch_size)
        ]

        batch_data_entities = [
            DetBatchDataEntity(
                batch_size=self.batch_size,
                images=tiles[i : i + self.batch_size],
                imgs_info=img_infos[i : i + self.batch_size],
                bboxes=[[] for _ in range(self.batch_size)],
                labels=[[] for _ in range(self.batch_size)],
            )
            for i in range(0, len(tiles), self.batch_size)
        ]
        return list(zip(batch_tile_attr_list, batch_data_entities))

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
                    raise TypeError(msg)

        return TileBatchDetDataEntity(
            batch_size=batch_size,
            batch_tiles=[[entity.image for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_img_infos=[[entity.img_info for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_tile_attr_list=[tile_entity.tile_attr_list for tile_entity in batch_entities],
            bboxes=[tile_entity.ori_bboxes for tile_entity in batch_entities],
            labels=[tile_entity.ori_labels for tile_entity in batch_entities],
        )


@dataclass
class TileInstSegDataEntity:
    """Data entity for tile task.

    :param entity: A list of OTXDataEntity
    """

    num_tiles: int
    entity_list: list[InstanceSegDataEntity]
    tile_attr_list: list[dict[str, int | str]]
    ori_bboxes: tv_tensors.BoundingBoxes
    ori_labels: LongTensor
    ori_masks: tv_tensors.Mask
    ori_polygons: list[Polygon]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.INSTANCE_SEGMENTATION


@dataclass
class TileBatchInstSegDataEntity(OTXTileBatchDataEntity):
    """Batch data entity for tile task."""

    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[LongTensor]
    masks: list[tv_tensors.Mask]
    polygons: list[list[Polygon]]

    def unbind(self) -> list[tuple[list[dict[str, int | str]], InstanceSegBatchDataEntity]]:
        """Unbind batch data entity for instance segmentation task."""
        tiles = [tile for tiles in self.batch_tiles for tile in tiles]
        tile_infos = [tile_info for tile_infos in self.batch_img_infos for tile_info in tile_infos]
        tile_attr_list = [tile_attr for tile_attrs in self.batch_tile_attr_list for tile_attr in tile_attrs]

        batch_tile_attr_list = [
            tile_attr_list[i : i + self.batch_size] for i in range(0, len(tile_attr_list), self.batch_size)
        ]
        batch_data_entities = [
            InstanceSegBatchDataEntity(
                batch_size=self.batch_size,
                images=tiles[i : i + self.batch_size],
                imgs_info=tile_infos[i : i + self.batch_size],
                bboxes=[[] for _ in range(self.batch_size)],
                labels=[[] for _ in range(self.batch_size)],
                masks=[[] for _ in range(self.batch_size)],
                polygons=[[] for _ in range(self.batch_size)],
            )
            for i in range(0, len(tiles), self.batch_size)
        ]
        return list(zip(batch_tile_attr_list, batch_data_entities))

    @classmethod
    def collate_fn(cls, batch_entities: list[TileInstSegDataEntity]) -> TileBatchInstSegDataEntity:
        """Collate function to collect TileInstSegDataEntity into TileBatchInstSegDataEntity in data loader."""
        if (batch_size := len(batch_entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        task = batch_entities[0].task

        for tile_entity in batch_entities:
            for entity in tile_entity.entity_list:
                if entity.task != task:
                    msg = "collate_fn() input should include a single OTX task"
                    raise RuntimeError(msg)

                if not isinstance(entity, InstanceSegDataEntity):
                    msg = "All entities should be InstanceSegDataEntity before collate_fn()"
                    raise TypeError(msg)

        return TileBatchInstSegDataEntity(
            batch_size=batch_size,
            batch_tiles=[[entity.image for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_img_infos=[[entity.img_info for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_tile_attr_list=[tile_entity.tile_attr_list for tile_entity in batch_entities],
            bboxes=[tile_entity.ori_bboxes for tile_entity in batch_entities],
            labels=[tile_entity.ori_labels for tile_entity in batch_entities],
            masks=[tile_entity.ori_masks for tile_entity in batch_entities],
            polygons=[tile_entity.ori_polygons for tile_entity in batch_entities],
        )
