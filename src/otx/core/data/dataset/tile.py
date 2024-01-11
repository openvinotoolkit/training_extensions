# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX tile dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from datumaro import Bbox, DatasetItem, DatasetSubset, Image, Polygon
from datumaro import Dataset as DmDataset
from datumaro.plugins.tiling import Tile
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.entity.instance_segmentation import InstanceSegDataEntity
from otx.core.data.entity.tile import (
    TileBatchDetDataEntity,
    TileBatchInstSegDataEntity,
    TileDetDataEntity,
    TileInstSegDataEntity,
)
from otx.core.types.task import OTXTaskType
from otx.core.utils.mask_util import polygon_to_bitmap

from .base import OTXDataset

if TYPE_CHECKING:
    from otx.core.config.data import TilerConfig
    from otx.core.data.dataset.detection import OTXDetectionDataset
    from otx.core.data.dataset.instance_segmentation import OTXInstanceSegDataset
    from otx.core.data.entity.base import OTXDataEntity

# ruff: noqa: SLF001
# NOTE: Disable private-member-access (SLF001).
# This is a workaround so we could apply the same transforms to tiles as the original dataset.


class OTXTileDatasetFactory:
    """OTX tile dataset factory."""

    @classmethod
    def create(
        cls,
        task: OTXTaskType,
        dataset: OTXDataset,
        tile_config: TilerConfig,
    ) -> OTXTileDataset:
        """Create a tile dataset based on the task type and subset type.

        NOte: All task utilize the same OTXTileTrainDataset for training.
              In testing, we use different tile dataset for different task
              type due to different annotation format and data entity.

        Args:
            task (OTXTaskType): OTX task type.
            dataset (OTXDataset): OTX dataset.
            tile_config (TilerConfig): Tile configuration.

        Returns:
            OTXTileDataset: Tile dataset.
        """
        if dataset.dm_subset.name == "train":
            return OTXTileTrainDataset(dataset, tile_config)

        if task == OTXTaskType.DETECTION:
            return OTXTileDetTestDataset(dataset, tile_config)
        if task == OTXTaskType.INSTANCE_SEGMENTATION:
            return OTXTileInstSegTestDataset(dataset, tile_config)
        msg = f"Unsupported task type: {task}"
        raise NotImplementedError(msg)


class OTXTileDataset(OTXDataset):
    """OTX tile dataset base class.

    Args:
        dataset (OTXDataset): OTX dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXDataset, tile_config: TilerConfig) -> None:
        super().__init__(
            dataset.dm_subset,
            dataset.transforms,
            dataset.mem_cache_handler,
            dataset.mem_cache_img_max_size,
            dataset.max_refetch,
        )
        self.tile_config = tile_config
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset.ids)

    @property
    def collate_fn(self) -> Callable:
        """Collate function from the original dataset."""
        return self._dataset.collate_fn

    def _get_item_impl(self, index: int) -> OTXDataEntity | None:
        """Get item implementation from the original dataset."""
        return self._dataset._get_item_impl(index)


class OTXTileTrainDataset(OTXTileDataset):
    """OTX tile train dataset.

    Args:
        dataset (OTXDataset): OTX dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXDataset, tile_config: TilerConfig) -> None:
        dm_dataset = dataset.dm_subset.as_dataset()
        dm_dataset = dm_dataset.transform(
            Tile,
            grid_size=tile_config.grid_size,
            overlap=(tile_config.overlap, tile_config.overlap),
            threshold_drop_ann=0.5,
        )
        dm_dataset = dm_dataset.filter("/item/annotation", filter_annotations=True, remove_empty=True)
        dm_subset = DatasetSubset(dm_dataset, dataset.dm_subset.name)
        dataset.dm_subset = dm_subset
        dataset.ids = [item.id for item in dm_subset]
        super().__init__(dataset, tile_config)


class OTXTileDetTestDataset(OTXTileDataset):
    """OTX tile detection test dataset.

    OTXTileDetTestDataset wraps a list of tiles (DetDataEntity) into a single TileDetDataEntity for testing/predicting.

    Args:
        dataset (OTXDetDataset): OTX detection dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXDetectionDataset, tile_config: TilerConfig) -> None:
        super().__init__(dataset, tile_config)

    @property
    def collate_fn(self) -> Callable:
        """Collate function for tile detection test dataset."""
        return TileBatchDetDataEntity.collate_fn

    def _get_item_impl(self, index: int) -> TileDetDataEntity:  # type: ignore[override]
        """Get item implementation.

        Transform a single dataset item to multiple tiles using Datumaro tiling plugin, and
        wrap tiles into a single TileDetDataEntity.

        Args:
            index (int): Index of the dataset item.

        Returns:
            TileDetDataEntity: tile detection data entity that wraps a list of detection data entities.

        Note:
            Ignoring [override] check is necessary here since OTXDataset._get_item_impl exclusively permits
            the return of OTXDataEntity. Nevertheless, in instances involving tiling, it becomes
            imperative to encapsulate tiles within a unified entity, namely TileDetDataEntity.
        """
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]

        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        # NOTE: transform could only be applied to DmDataset and not directly to DatasetItem
        tile_ds = DmDataset.from_iterable([item])
        tile_ds = tile_ds.transform(
            Tile,
            grid_size=self.tile_config.grid_size,
            overlap=(self.tile_config.overlap, self.tile_config.overlap),
            threshold_drop_ann=0.5,
        )
        tile_entities: list[DetDataEntity] = []
        for tile in tile_ds:
            tile_entity = self._convert_entity(tile)
            # apply the same transforms as the original dataset
            transformed_tile = self._apply_transforms(tile_entity)
            if transformed_tile is None:
                msg = "Transformed tile is None"
                raise RuntimeError(msg)
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

    def _convert_entity(self, dataset_item: DatasetItem) -> DetDataEntity:
        """Convert a tile datumaro dataset item to DetDataEntity."""
        tile_img = dataset_item.media_as(Image).data
        tile_shape = tile_img.shape[:2]
        img_info = ImageInfo(
            img_idx=dataset_item.attributes["id"],
            img_shape=tile_shape,
            ori_shape=tile_shape,
            attributes=dataset_item.attributes,
        )
        return DetDataEntity(
            image=tile_img,
            img_info=img_info,
            # we don't need tile-level annotations
            bboxes=tv_tensors.BoundingBoxes(
                [],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=tile_shape,
            ),
            labels=torch.as_tensor([]),
        )


class OTXTileInstSegTestDataset(OTXTileDataset):
    """OTX tile inst-seg test dataset.

    OTXTileDetTestDataset wraps a list of tiles (InstanceSegDataEntity) into a single TileDetDataEntity
    for testing/predicting.

    Args:
        dataset (OTXInstanceSegDataset): OTX inst-seg dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXInstanceSegDataset, tile_config: TilerConfig) -> None:
        super().__init__(dataset, tile_config)

    @property
    def collate_fn(self) -> Callable:
        """Collate function for tile inst-seg test dataset."""
        return TileBatchInstSegDataEntity.collate_fn

    def _get_item_impl(self, index: int) -> TileInstSegDataEntity:  # type: ignore[override]
        """Get item implementation.

        Transform a single dataset item to multiple tiles using Datumaro tiling plugin, and
        wrap tiles into a single TileInstSegDataEntity.

        Args:
            index (int): Index of the dataset item.

        Returns:
            TileInstSegDataEntity: tile inst-seg data entity that wraps a list of inst-seg data entities.

        Note:
            Ignoring [override] check is necessary here since OTXDataset._get_item_impl exclusively permits
            the return of OTXDataEntity. Nevertheless, in instances involving tiling, it becomes
            imperative to encapsulate tiles within a unified entity, namely TileInstSegDataEntity.
        """
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        gt_bboxes, gt_labels, gt_masks, gt_polygons = [], [], [], []

        for annotation in item.annotations:
            if isinstance(annotation, Polygon):
                bbox = np.array(annotation.get_bbox(), dtype=np.float32)
                gt_bboxes.append(bbox)
                gt_labels.append(annotation.label)

                if self._dataset.include_polygons:
                    gt_polygons.append(annotation)
                else:
                    gt_masks.append(polygon_to_bitmap([annotation], *img_shape)[0])

        # convert xywh to xyxy format
        bboxes = np.array(gt_bboxes, dtype=np.float32)
        bboxes[:, 2:] += bboxes[:, :2]

        masks = np.stack(gt_masks, axis=0) if gt_masks else np.zeros((0, *img_shape), dtype=bool)
        labels = np.array(gt_labels, dtype=np.int64)

        # NOTE: transform could only be applied to DmDataset and not directly to DatasetItem
        tile_ds = DmDataset.from_iterable([item])
        tile_ds = tile_ds.transform(
            Tile,
            grid_size=self.tile_config.grid_size,
            overlap=(self.tile_config.overlap, self.tile_config.overlap),
            threshold_drop_ann=0.5,
        )
        tile_entities: list[InstanceSegDataEntity] = []
        for tile in tile_ds:
            tile_entity = self._convert_entity(tile)
            # apply the same transforms as the original dataset
            transformed_tile = self._apply_transforms(tile_entity)
            if transformed_tile is None:
                msg = "Transformed tile is None"
                raise RuntimeError(msg)
            tile_entities.append(transformed_tile)

        return TileInstSegDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            ori_bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            ori_labels=torch.as_tensor(labels),
            ori_masks=tv_tensors.Mask(masks, dtype=torch.uint8),
            ori_polygons=gt_polygons,
        )

    def _convert_entity(self, dataset_item: DatasetItem) -> InstanceSegDataEntity:
        """Convert a tile dataset item to InstanceSegDataEntity."""
        tile_img = dataset_item.media_as(Image).data
        tile_shape = tile_img.shape[:2]
        img_info = ImageInfo(
            img_idx=dataset_item.attributes["id"],
            img_shape=tile_shape,
            ori_shape=tile_shape,
            attributes=dataset_item.attributes,
        )
        return InstanceSegDataEntity(
            image=tile_img,
            img_info=img_info,
            # we don't need tile-level annotations
            bboxes=tv_tensors.BoundingBoxes(
                [],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=tile_shape,
            ),
            labels=torch.as_tensor([]),
            masks=tv_tensors.Mask(np.zeros((0, *tile_shape), dtype=bool)),
            polygons=[],
        )
