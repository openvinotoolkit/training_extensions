# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import numpy as np
import pytest
from datumaro import Dataset as DmDataset
from omegaconf import DictConfig, OmegaConf
from otx.core.config.data import (
    DataModuleConfig,
    SubsetConfig,
    TileConfig,
    VisualPromptingConfig,
)
from otx.core.data.dataset.tile import OTXTileTransform
from otx.core.data.entity.detection import DetBatchDataEntity
from otx.core.data.entity.tile import TileBatchDetDataEntity
from otx.core.data.module import OTXDataModule
from otx.core.types.task import OTXTaskType


class TestOTXTiling:
    @pytest.fixture()
    def fxt_mmcv_det_transform_config(self) -> list[DictConfig]:
        mmdet_base = OmegaConf.load("src/otx/recipe/_base_/data/mmdet_base.yaml")
        return mmdet_base.config.train_subset.transforms

    @pytest.fixture()
    def fxt_det_data_config(self, fxt_asset_dir, fxt_mmcv_det_transform_config) -> OTXDataModule:
        data_root = fxt_asset_dir / "car_tree_bug"

        batch_size = 8
        num_workers = 0
        return DataModuleConfig(
            data_format="coco_instances",
            data_root=data_root,
            train_subset=SubsetConfig(
                subset_name="train",
                batch_size=batch_size,
                num_workers=num_workers,
                transform_lib_type="MMDET",
                transforms=fxt_mmcv_det_transform_config,
            ),
            val_subset=SubsetConfig(
                subset_name="val",
                batch_size=batch_size,
                num_workers=num_workers,
                transform_lib_type="MMDET",
                transforms=fxt_mmcv_det_transform_config,
            ),
            test_subset=SubsetConfig(
                subset_name="test",
                batch_size=batch_size,
                num_workers=num_workers,
                transform_lib_type="MMDET",
                transforms=fxt_mmcv_det_transform_config,
            ),
            tile_config=TileConfig(),
            vpm_config=VisualPromptingConfig(),
        )

    def test_tile_transform(self):
        dataset = DmDataset.import_from("tests/assets/car_tree_bug", format="coco_instances")
        first_item = next(iter(dataset), None)
        height, width = first_item.media.data.shape[:2]

        rng = np.random.default_rng()
        tile_size = rng.integers(low=100, high=500, size=(2,))
        overlap = rng.random(2)
        threshold_drop_ann = rng.random()
        tiled_dataset = DmDataset.import_from("tests/assets/car_tree_bug", format="coco_instances")
        tiled_dataset.transform(
            OTXTileTransform,
            tile_size=tile_size,
            overlap=overlap,
            threshold_drop_ann=threshold_drop_ann,
        )

        h_stride = int((1 - overlap[0]) * tile_size[0])
        w_stride = int((1 - overlap[1]) * tile_size[1])
        num_tile_rows = (height + h_stride - 1) // h_stride
        num_tile_cols = (width + w_stride - 1) // w_stride
        assert len(tiled_dataset) == (num_tile_rows * num_tile_cols * len(dataset)), "Incorrect number of tiles"

    def test_adaptive_tiling(self, fxt_det_data_config):
        # Enable tile adapter
        fxt_det_data_config.tile_config.enable_tiler = True
        fxt_det_data_config.tile_config.enable_adaptive_tiling = True
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            config=fxt_det_data_config,
        )
        tile_datamodule.prepare_data()

        assert tile_datamodule.config.tile_config.tile_size == (6750, 6750), "Tile size should be [6750, 6750]"
        assert (
            pytest.approx(tile_datamodule.config.tile_config.overlap, rel=1e-3) == 0.03608
        ), "Overlap should be 0.03608"
        assert tile_datamodule.config.tile_config.max_num_instances == 3, "Max num instances should be 3"

    def test_tile_sampler(self, fxt_det_data_config):
        rng = np.random.default_rng()

        fxt_det_data_config.tile_config.enable_tiler = True
        fxt_det_data_config.tile_config.enable_adaptive_tiling = False
        fxt_det_data_config.tile_config.sampling_ratio = rng.random()
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            config=fxt_det_data_config,
        )
        tile_datamodule.prepare_data()
        sampled_count = max(
            1,
            int(len(tile_datamodule._get_dataset("train")) * fxt_det_data_config.tile_config.sampling_ratio),
        )

        count = 0
        for batch in tile_datamodule.train_dataloader():
            count += batch.batch_size
            assert isinstance(batch, DetBatchDataEntity)

        assert sampled_count == count, "Sampled count should be equal to the count of the dataloader batch size"

    def test_train_dataloader(self, fxt_det_data_config) -> None:
        # Enable tile adapter
        fxt_det_data_config.tile_config.enable_tiler = True
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            config=fxt_det_data_config,
        )
        tile_datamodule.prepare_data()
        for batch in tile_datamodule.train_dataloader():
            assert isinstance(batch, DetBatchDataEntity)

    def test_val_dataloader(self, fxt_det_data_config) -> None:
        # Enable tile adapter
        fxt_det_data_config.tile_config.enable_tiler = True
        tile_datamodule = OTXDataModule(
            task=OTXTaskType.DETECTION,
            config=fxt_det_data_config,
        )
        tile_datamodule.prepare_data()
        for batch in tile_datamodule.val_dataloader():
            assert isinstance(batch, TileBatchDetDataEntity)

    def test_tile_merge(self):
        pytest.skip("Not implemented yet")
