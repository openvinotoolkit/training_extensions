# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX tile dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from datumaro import Bbox, DatasetSubset, Image
from datumaro import Dataset as DmDataset
from datumaro.plugins.tiling import Tile
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.entity.tile import TileBatchDetDataEntity, TileDetDataEntity

from .base import OTXDataset

if TYPE_CHECKING:
    from datumaro import DatasetItem

    from otx.core.config.data import TilerConfig
    from otx.core.data.entity.base import OTXDataEntity


class OTXTileTrainDataset(OTXDataset):
    def __init__(self, dataset: OTXDataset, tile_config: TilerConfig) -> None:
        # Preprocess the dataset for training with tiling
        dm_dataset = dataset.dm_subset.as_dataset()
        dm_dataset = dm_dataset.transform(
            Tile,
            grid_size=tile_config.grid_size,
            overlap=(tile_config.overlap, tile_config.overlap),
            threshold_drop_ann=0.5,
        )
        dm_dataset = dm_dataset.filter("/item/annotation", filter_annotations=True, remove_empty=True)
        dm_subset = DatasetSubset(dm_dataset, dataset.dm_subset.name)
        super().__init__(dm_subset, transforms=dataset.transforms)
        OTXTileTrainDataset._get_item_impl = dataset.__class__._get_item_impl
        OTXTileTrainDataset.collate_fn = dataset.__class__.collate_fn


class OTXTileTestDataset(OTXDataset):
    def __init__(self, dataset: OTXDataset, tile_config: TilerConfig) -> None:
        # Initialize the test dataset with tiling configuration
        self.tile_config = tile_config
        super().__init__(dataset.dm_subset, transforms=dataset.transforms)

    @property
    def collate_fn(self) -> Callable:
        return TileBatchDetDataEntity.collate_fn

    def _get_item_impl(self, index: int) -> TileDetDataEntity | None:
        # Retrieve a dataset item from the subset
        dataset_item = self.dm_subset.get(
            self.ids[index],
            subset=self.dm_subset.name,
        )
        bbox_anns = [ann for ann in dataset_item.annotations if isinstance(ann, Bbox)]

        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        # Extract image information and apply tiling transformation
        img = dataset_item.media_as(Image)
        _, img_shape = self._get_img_data_and_shape(img)

        # NOTE: transform could only be applied to DmDataset and not directly to DatasetItem
        tile_ds = DmDataset.from_iterable([dataset_item])
        tile_ds = tile_ds.transform(
            Tile,
            grid_size=self.tile_config.grid_size,
            overlap=(self.tile_config.overlap, self.tile_config.overlap),
            threshold_drop_ann=0.5,
        )
        tile_entities = []
        for tile in tile_ds:
            tile_entity = self.convert_tile(tile)
            # apply the same transforms as the original dataset
            transformed_tile = self._apply_transforms(tile_entity)
            tile_entities.append(transformed_tile)

        return TileDetDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            ori_bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            ori_labels=torch.as_tensor([ann.label for ann in bbox_anns]),
        )

    def convert_tile(self, tile_item: DatasetItem) -> OTXDataEntity:
        """Convert a tile dataset item to OTXDataEntity.

        Args:
            tile_item (DatasetItem): tile dataset item

        Returns:
            OTXDataEntity: OTXDataEntity
        """
        tile_img = tile_item.media_as(Image).data
        tile_shape = tile_img.shape[:2]

        return DetDataEntity(
            image=tile_img,
            img_info=ImageInfo(
                img_idx=tile_item.attributes["id"],
                img_shape=tile_shape,
                ori_shape=tile_shape,
                pad_shape=tile_shape,
                scale_factor=(1.0, 1.0),
                attributes=tile_item.attributes,
            ),
            # we don't need tile-level annotations
            bboxes=tv_tensors.BoundingBoxes(
                [],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=tile_shape,
            ),
            labels=torch.as_tensor([]),
        )
