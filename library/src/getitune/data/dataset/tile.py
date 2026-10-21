# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""getitune tile dataset."""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Callable

import torch
from datumaro.experimental.fields import Subset
from datumaro.experimental.fields.datasets import TileInfo
from datumaro.experimental.filtering.filter_registry import create_filtering_transform
from datumaro.experimental.tiling.tiler_registry import TilingConfig, create_tiling_transform
from torchvision import tv_tensors

from getitune.data.entity.sample import BaseSample, DetectionSample, InstanceSegmentationSample
from getitune.data.entity.tile import (
    TileBatchDetDataEntity,
    TileBatchInstSegDataEntity,
    TileBatchSegDataEntity,
    TileDetDataEntity,
    TileInstSegDataEntity,
    TileSegDataEntity,
)
from getitune.data.utils.utils import fill_null_annotation_lists as _fill_null_annotation_lists
from getitune.types.task import TaskType

from .base import VisionDataset, _ensure_chw_format

if TYPE_CHECKING:
    from getitune.config.data import TileConfig
    from getitune.data.dataset.detection import DetectionDataset
    from getitune.data.dataset.instance_segmentation import InstanceSegDataset
    from getitune.data.dataset.segmentation import SegmentationDataset


# ruff: noqa: SLF001
# NOTE: Disable private-member-access (SLF001).
# This is a workaround so we could apply the same transforms to tiles as the original dataset.


class TileDatasetFactory:
    """getitune tile dataset factory."""

    @classmethod
    def create(
        cls,
        dataset: VisionDataset,
        tile_config: TileConfig,
    ) -> VisionDataset:
        """Create a tile dataset based on the task type and subset type.

        NOte: All task utilize the same TileTrainDataset for training.
              In testing, we use different tile dataset for different task
              type due to different annotation format and data entity.

        Args:
            task (TaskType): getitune task type.
            dataset (VisionDataset): getitune dataset.
            tile_config (TilerConfig): Tile configuration.

        Returns:
            TileDataset: Tile dataset.
        """
        subset = dataset.dm_subset[0].subset  # type: ignore[attr-defined]
        if subset == Subset.TRAINING:
            dm_dataset = dataset.dm_subset
            dm_dataset = dm_dataset.transform(  # type: ignore[attr-defined]
                create_tiling_transform(
                    TilingConfig(
                        tile_height=tile_config.tile_size[0],
                        tile_width=tile_config.tile_size[1],
                        overlap_x=tile_config.overlap,
                        overlap_y=tile_config.overlap,
                    ),
                    threshold_drop_ann=0.5,
                ),
                dtype=dm_dataset.dtype,  # type: ignore[attr-defined]
            )
            dm_dataset = dm_dataset.transform(create_filtering_transform(), dtype=dm_dataset.dtype)  # type: ignore[attr-defined]
            dataset.dm_subset = dm_dataset
            return dataset

        if dataset.task_type == TaskType.DETECTION:
            return TileDetTestDataset(dataset, tile_config, subset)
        if dataset.task_type == TaskType.INSTANCE_SEGMENTATION:
            return TileInstSegTestDataset(dataset, tile_config, subset)
        if dataset.task_type == TaskType.SEMANTIC_SEGMENTATION:
            return TileSemanticSegTestDataset(dataset, tile_config, subset)

        msg = f"Unsupported task type: {dataset.task_type} for tiling"
        raise NotImplementedError(msg)


class TileDataset(VisionDataset):
    """getitune tile dataset base class.

    Args:
        dataset (VisionDataset): getitune dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: VisionDataset, tile_config: TileConfig, subset: Subset) -> None:
        super().__init__(
            dataset.dm_subset,
            dataset.transforms,
            dataset.max_refetch,
        )
        # Sanitize once up front: `self.dm_subset` is fixed for the lifetime of this
        # dataset, and `.slice(...)` on an already-null-filled dataframe never
        # reintroduces nulls, so there is no need to re-sanitize on every
        # `get_tiles()` call (which runs once per image, per epoch, during eval).
        self.dm_subset = _fill_null_annotation_lists(self.dm_subset)
        self.tile_config = tile_config
        self._dataset = dataset
        self._subset = subset
        self._tiling_config = TilingConfig(
            tile_height=tile_config.tile_size[0],
            tile_width=tile_config.tile_size[1],
            overlap_x=tile_config.overlap,
            overlap_y=tile_config.overlap,
        )
        self._tiling_transform = create_tiling_transform(self._tiling_config, threshold_drop_ann=0.5)
        self._filtering_transform = create_filtering_transform()

        # LabelInfo differs from SegLabelInfo, thus we need to update it for semantic segmentation.
        if self.label_info != dataset.label_info:
            msg = (
                "Replace the label info to match the dataset's label info",
                "as there is a mismatch between the dataset and the tile dataset.",
            )
            log.warning(msg)
            self.label_info = dataset.label_info

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def collate_fn(self) -> Callable:
        """Collate function from the original dataset."""
        return self._dataset.collate_fn

    def _get_item_impl(self, index: int) -> BaseSample | None:
        """Get item implementation from the original dataset."""
        return self._dataset._get_item_impl(index)

    def get_tiles(
        self,
        parent_idx: int,
    ) -> list[BaseSample]:
        """Retrieves tiles from the given image and dataset item.

        Args:
            image (np.ndarray): The input image.
            item (DatasetItem): The dataset item.
            parent_idx (int): The parent index. This is to keep track of the original dataset item index for merging.

        Returns:
            A tuple containing two lists:
            - tile_entities (list[BaseSample]): List of tile entities.
        """
        parent_slice_ds = self.dm_subset.slice(parent_idx, 1)  # type: ignore[attr-defined]
        tile_ds = parent_slice_ds.transform(  # type: ignore[attr-defined]
            self._tiling_transform,
            dtype=parent_slice_ds.dtype,  # type: ignore[attr-defined]
        )

        if self._subset == Subset.VALIDATION:
            # NOTE: filter validation tiles with annotations only to avoid evaluation on empty tiles.
            tile_ds = tile_ds.transform(self._filtering_transform, dtype=parent_slice_ds.dtype)  # type: ignore[attr-defined]

            # if tile dataset is empty (or the filtering transform returned None because the
            # source image has no annotations at all) it means objects are too big to fit in
            # any tile, or there are no objects to fit; in this case include the full image.
            if tile_ds is None or len(tile_ds) == 0:
                # No tiles produced (image too small, or filtering removed all tiles
                # from a background image with no annotations). Create a single
                # synthetic "tile" that covers the full image so the downstream tile
                # merger / unbind logic still sees one tile per source image (keeps
                # ``img_infos`` and ``img_ids`` lengths consistent).
                tile_entities: list[BaseSample] = []
                for raw_item in parent_slice_ds:
                    raw_item.image = _ensure_chw_format(raw_item.image)
                    img = raw_item.img_info
                    tile_info = TileInfo(
                        source_sample_idx=parent_idx,
                        x=0,
                        y=0,
                        width=img.img_shape[1],
                        height=img.img_shape[0],
                    )
                    object.__setattr__(raw_item, "tile", tile_info)
                    transformed = self._apply_transforms(raw_item)
                    if transformed is None:
                        continue
                    # Undo transform corruption of empty tensors (e.g. `Resize`
                    # turns ``(0,4)`` into ``(1,0)``). Only detection / instance
                    # segmentation samples carry `bboxes`/`label` fields.
                    if (
                        isinstance(transformed, (DetectionSample, InstanceSegmentationSample))
                        and transformed.label is not None
                        and len(transformed.label) == 0
                    ):
                        shape = transformed.img_info.img_shape
                        transformed.bboxes = tv_tensors.BoundingBoxes(
                            torch.zeros((0, 4), dtype=torch.float32),
                            format=tv_tensors.BoundingBoxFormat.XYXY,
                            canvas_size=shape,
                            dtype=torch.float32,
                        )
                        # Keep the sample's label dtype so later concatenation (e.g. Mosaic/MixUp)
                        # does not mix dtypes; torch cannot promote uint16 with other integer types.
                        transformed.label = torch.zeros(0, dtype=transformed.label.dtype)
                    tile_entities.append(transformed)
                return tile_entities

        tile_entities: list[BaseSample] = []
        for tile in tile_ds:
            # Fix datumaro HWC→CHW format issue for tile images
            tile.image = _ensure_chw_format(tile.image)
            # apply the same transforms as the original dataset
            object.__setattr__(tile.tile, "source_sample_idx", parent_idx)
            transformed_tile = self._apply_transforms(tile)
            if transformed_tile is None:
                msg = "Transformed tile is None"
                raise RuntimeError(msg)
            tile_entities.append(transformed_tile)
        return tile_entities


class TileDetTestDataset(TileDataset):
    """getitune tile detection test dataset.

    TileDetTestDataset wraps a list of tiles (DetDataEntity) into a single TileDetDataEntity for testing/predicting.

    Args:
        dataset (DetectionDataset): getitune detection dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: DetectionDataset, tile_config: TileConfig, subset: Subset) -> None:
        super().__init__(dataset, tile_config, subset)

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
            Ignoring [override] check is necessary here since VisionDataset._get_item_impl exclusively permits
            the return of BaseSample. Nevertheless, in instances involving tiling, it becomes
            imperative to encapsulate tiles within a unified entity, namely TileDetDataEntity.
        """
        item = self.dm_subset[index]
        tile_entities = self.get_tiles(index)

        return TileDetDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            ori_img_info=item.img_info,  # type: ignore[attr-defined]
            ori_bboxes=item.bboxes,  # type: ignore[attr-defined]
            ori_labels=item.label,  # type: ignore[attr-defined]
        )


class TileInstSegTestDataset(TileDataset):
    """getitune tile inst-seg test dataset.

    TileDetTestDataset wraps a list of tiles (TorchDataItem) into a single TileDetDataEntity
    for testing/predicting.

    Args:
        dataset (InstanceSegDataset): getitune inst-seg dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: InstanceSegDataset, tile_config: TileConfig, subset: Subset) -> None:
        super().__init__(dataset, tile_config, subset)

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
            Ignoring [override] check is necessary here since VisionDataset._get_item_impl exclusively permits
            the return of BaseSample. Nevertheless, in instances involving tiling, it becomes
            imperative to encapsulate tiles within a unified entity, namely TileInstSegDataEntity.
        """
        item = self.dm_subset[index]
        tile_entities = self.get_tiles(index)

        return TileInstSegDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            ori_img_info=item.img_info,  # type: ignore[attr-defined]
            ori_bboxes=item.bboxes,  # type: ignore[attr-defined]
            ori_labels=item.label,  # type: ignore[attr-defined]
            ori_masks=item.masks,  # type: ignore[attr-defined]
        )


class TileSemanticSegTestDataset(TileDataset):
    """getitune tile semantic-seg test dataset.

    TileSemanticSegTestDataset wraps a list of tiles (SegDataEntity) into a single TileSegDataEntity
    for testing/predicting.

    Args:
        dataset (SegmentationDataset): getitune semantic-seg dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: SegmentationDataset, tile_config: TileConfig, subset: Subset) -> None:
        super().__init__(dataset, tile_config, subset)

    @property
    def collate_fn(self) -> Callable:
        """Collate function for tile detection test dataset."""
        return TileBatchSegDataEntity.collate_fn

    def _get_item_impl(self, index: int) -> TileSegDataEntity:  # type: ignore[override]
        """Get item implementation.

        Transform a single dataset item to multiple tiles using Datumaro tiling plugin, and
        wrap tiles into a single TileSegDataEntity.

        Args:
            index (int): Index of the dataset item.

        Returns:
            TileSegDataEntity: tile semantic-seg data entity that wraps a list of semantic-seg data entities.
        """
        item = self.dm_subset[index]
        tile_entities = self.get_tiles(index)

        return TileSegDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            ori_img_info=item.img_info,  # type: ignore[attr-defined]
            ori_masks=item.masks,  # type: ignore[attr-defined]
        )
