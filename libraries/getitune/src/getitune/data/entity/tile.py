# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for getitune tile data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from torchvision import tv_tensors

from getitune.data.entity.sample import BaseSample, SampleBatch
from getitune.data.entity.utils import stack_batch
from getitune.types.task import TaskType

from .base import ImageInfo

if TYPE_CHECKING:
    from datumaro.experimental.fields import TileInfo
    from torch import LongTensor


@dataclass
class TileDataEntity:
    """Base data entity for tile task.

    Attributes:
        num_tiles (int): The number of tiles.
        entity_list (Sequence[DataEntity]): A list of DataEntity.
        ori_img_info (ImageInfo): The image information about the original image.
    """

    num_tiles: int
    entity_list: Sequence[BaseSample]
    ori_img_info: ImageInfo

    @property
    def task(self) -> TaskType:
        """Getitune Task type definition."""
        raise NotImplementedError


@dataclass
class TileDetDataEntity(TileDataEntity):
    """Data entity for detection tile task.

    Attributes:
        ori_bboxes (tv_tensors.BoundingBoxes): The bounding boxes of the original image.
        ori_labels (LongTensor): The labels of the original image.
    """

    ori_bboxes: tv_tensors.BoundingBoxes
    ori_labels: LongTensor

    @property
    def task(self) -> TaskType:
        """Getitune Task type definition."""
        return TaskType.DETECTION


TileAttrDictList = list[dict[str, int | str]]


@dataclass
class TileBatchData:
    """Base batch data entity for tile task.

    Attributes:
        batch_size (int): The size of the batch.
        batch_tiles (list[list[tv_tensors.Image]]): The batch of tile images.
        batch_tile_img_infos (list[list[ImageInfo]]): The batch of tiles image information.
        imgs_info (list[ImageInfo]): The image information about the original image.
    """

    batch_size: int
    batch_tiles: list[list[tv_tensors.Image]]
    batch_tile_img_infos: list[list[ImageInfo]]
    batch_tile_tile_infos: list[list[TileInfo]]
    imgs_info: list[ImageInfo]

    def unbind(self, tile_batch_size: int) -> list[tuple[list[TileInfo], SampleBatch]]:
        """Unbind batch data entity.

        Args:
            tile_batch_size: Number of tiles grouped into a single model forward pass.
        """
        raise NotImplementedError


@dataclass
class TileBatchDetDataEntity(TileBatchData):
    """Batch data entity for detection tile task.

    Attributes:
        bboxes (list[tv_tensors.BoundingBoxes]): The bounding boxes of the original image.
        labels (list[LongTensor]): The labels of the original image.
    """

    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[LongTensor]

    def unbind(self, tile_batch_size: int) -> list[tuple[list[TileInfo], SampleBatch]]:
        """Unbind batch data entity for detection task."""
        tiles = [tile for tiles in self.batch_tiles for tile in tiles]
        tile_img_infos = [tile_info for tile_infos in self.batch_tile_img_infos for tile_info in tile_infos]
        tile_tile_infos = [tile_info for tile_infos in self.batch_tile_tile_infos for tile_info in tile_infos]

        batch_data_entities = []
        for i in range(0, len(tiles), tile_batch_size):
            stacked_images, updated_img_info = stack_batch(
                tiles[i : i + tile_batch_size],
                tile_img_infos[i : i + tile_batch_size],
            )
            batch_data_entities.append(
                (
                    tile_tile_infos[i : i + tile_batch_size],
                    SampleBatch(
                        images=stacked_images,
                        imgs_info=updated_img_info,
                    ),
                )
            )
        return batch_data_entities

    @classmethod
    def collate_fn(cls, batch_entities: list[TileDetDataEntity]) -> TileBatchDetDataEntity:
        """Collate function to collect TileDetDataEntity into TileBatchDetDataEntity in data loader."""
        if (batch_size := len(batch_entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        for tile_entity in batch_entities:
            for entity in tile_entity.entity_list:
                if not isinstance(entity, BaseSample):
                    msg = "All entities should be BaseSample before collate_fn()"
                    raise TypeError(msg)
                if entity.img_info is None:
                    msg = "All entities should have img_info, but found None"
                    raise ValueError(msg)

        return TileBatchDetDataEntity(
            batch_size=batch_size,
            batch_tiles=[[entity.image for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_tile_img_infos=[
                [entity.img_info for entity in tile_entity.entity_list] for tile_entity in batch_entities
            ],
            batch_tile_tile_infos=[
                [entity.tile for entity in tile_entity.entity_list] for tile_entity in batch_entities
            ],
            imgs_info=[tile_entity.ori_img_info for tile_entity in batch_entities],
            bboxes=[tile_entity.ori_bboxes for tile_entity in batch_entities],
            labels=[tile_entity.ori_labels for tile_entity in batch_entities],
        )


@dataclass
class TileInstSegDataEntity(TileDataEntity):
    """Data entity for instance segmentation tile task.

    Attributes:
        ori_bboxes (tv_tensors.BoundingBoxes): The bounding boxes of the original image.
        ori_labels (LongTensor): The labels of the original image.
        ori_masks (tv_tensors.Mask): The masks of the original image.
    """

    ori_bboxes: tv_tensors.BoundingBoxes
    ori_labels: LongTensor
    ori_masks: tv_tensors.Mask

    @property
    def task(self) -> TaskType:
        """Getitune Task type definition."""
        return TaskType.INSTANCE_SEGMENTATION


@dataclass
class TileBatchInstSegDataEntity(TileBatchData):
    """Batch data entity for instance segmentation tile task.

    Attributes:
        bboxes (list[tv_tensors.BoundingBoxes]): The bounding boxes of the original image.
        labels (list[LongTensor]): The labels of the original image.
        masks (list[tv_tensors.Mask]): The masks of the original image.
    """

    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[LongTensor]
    masks: list[tv_tensors.Mask]

    def unbind(self, tile_batch_size: int) -> list[tuple[TileAttrDictList, SampleBatch]]:
        """Unbind batch data entity for instance segmentation task."""
        tiles = [tile for tiles in self.batch_tiles for tile in tiles]
        tile_img_infos = [tile_info for tile_infos in self.batch_tile_img_infos for tile_info in tile_infos]
        tile_tile_infos = [tile_info for tile_infos in self.batch_tile_tile_infos for tile_info in tile_infos]

        batch_data_entities = [
            (
                tile_tile_infos[i : i + tile_batch_size],
                SampleBatch(
                    images=tiles[i : i + tile_batch_size],
                    imgs_info=tile_img_infos[i : i + tile_batch_size],
                ),
            )
            for i in range(0, len(tiles), tile_batch_size)
        ]
        return list(batch_data_entities)

    @classmethod
    def collate_fn(cls, batch_entities: list[TileInstSegDataEntity]) -> TileBatchInstSegDataEntity:
        """Collate function to collect TileInstSegDataEntity into TileBatchInstSegDataEntity in data loader."""
        if (batch_size := len(batch_entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        for tile_entity in batch_entities:
            for entity in tile_entity.entity_list:
                if not isinstance(entity, BaseSample):
                    msg = "All entities should be BaseSample before collate_fn()"
                    raise TypeError(msg)
                if entity.img_info is None:
                    msg = "All entities should have img_info, but found None"
                    raise ValueError(msg)

        return TileBatchInstSegDataEntity(
            batch_size=batch_size,
            batch_tiles=[[entity.image for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_tile_img_infos=[
                [entity.img_info for entity in tile_entity.entity_list if isinstance(entity.img_info, ImageInfo)]
                for tile_entity in batch_entities
            ],
            batch_tile_tile_infos=[
                [entity.tile for entity in tile_entity.entity_list] for tile_entity in batch_entities
            ],
            imgs_info=[tile_entity.ori_img_info for tile_entity in batch_entities],
            bboxes=[tile_entity.ori_bboxes for tile_entity in batch_entities],
            labels=[tile_entity.ori_labels for tile_entity in batch_entities],
            masks=[tile_entity.ori_masks for tile_entity in batch_entities],
        )


@dataclass
class TileSegDataEntity(TileDataEntity):
    """Data entity for segmentation tile task.

    Attributes:
        ori_masks (tv_tensors.Mask): The masks of the original image.
    """

    ori_masks: tv_tensors.Mask

    @property
    def task(self) -> TaskType:
        """Getitune Task type definition."""
        return TaskType.SEMANTIC_SEGMENTATION


@dataclass
class TileBatchSegDataEntity(TileBatchData):
    """Batch data entity for semantic segmentation tile task.

    Attributes:
        masks (list[tv_tensors.Mask]): The masks of the original image.
    """

    masks: list[tv_tensors.Mask]

    def unbind(self, tile_batch_size: int) -> list[tuple[list[dict[str, int | str]], SampleBatch]]:
        """Unbind batch data entity for semantic segmentation task."""
        tiles = [tile for tiles in self.batch_tiles for tile in tiles]
        tile_img_infos = [tile_info for tile_infos in self.batch_tile_img_infos for tile_info in tile_infos]
        tile_tile_infos = [tile_info for tile_infos in self.batch_tile_tile_infos for tile_info in tile_infos]

        batch_data_entities = [
            (
                tile_tile_infos[i : i + tile_batch_size],
                SampleBatch(
                    images=tv_tensors.wrap(
                        torch.stack(tiles[i : i + tile_batch_size]),  # pyrefly: ignore[bad-argument-type]
                        like=tiles[0],  # pyrefly: ignore[unexpected-keyword]
                    ),
                    imgs_info=tile_img_infos[i : i + tile_batch_size],
                    masks=[torch.empty((1, 1, 1)) for _ in range(len(tiles[i : i + tile_batch_size]))],
                ),
            )
            for i in range(0, len(tiles), tile_batch_size)
        ]
        return list(batch_data_entities)

    @classmethod
    def collate_fn(cls, batch_entities: list[TileSegDataEntity]) -> TileBatchSegDataEntity:
        """Collate function to collect TileSegDataEntity into TileBatchSegDataEntity in data loader."""
        if (batch_size := len(batch_entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        for tile_entity in batch_entities:
            for entity in tile_entity.entity_list:
                if not isinstance(entity, BaseSample):
                    msg = "All entities should be BaseSample before collate_fn()"
                    raise TypeError(msg)
                if entity.img_info is None:
                    msg = "All entities should have img_info, but found None"
                    raise ValueError(msg)

        return TileBatchSegDataEntity(
            batch_size=batch_size,
            batch_tiles=[[entity.image for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_tile_img_infos=[
                [entity.img_info for entity in tile_entity.entity_list] for tile_entity in batch_entities
            ],
            batch_tile_tile_infos=[
                [entity.tile for entity in tile_entity.entity_list] for tile_entity in batch_entities
            ],
            imgs_info=[tile_entity.ori_img_info for tile_entity in batch_entities],
            masks=[tile_entity.ori_masks for tile_entity in batch_entities],
        )
