# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from datumaro import Dataset as DmDataset
from datumaro import Image
from datumaro.plugins.tiling.util import xywh_to_x1y1x2y2
from omegaconf import DictConfig, OmegaConf
from openvino.model_api.models import Model
from openvino.model_api.tilers import DetectionTiler, InstanceSegmentationTiler, Tiler
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
from otx.core.model.entity.detection import OVDetectionModel
from otx.core.model.entity.instance_segmentation import OVInstanceSegmentationModel
from otx.core.types.task import OTXTaskType
from otx.engine import Engine


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

        h_stride = max(int((1 - overlap[0]) * tile_size[0]), 1)
        w_stride = max(int((1 - overlap[1]) * tile_size[1]), 1)
        num_tile_rows = (height + h_stride - 1) // h_stride
        num_tile_cols = (width + w_stride - 1) // w_stride
        assert len(tiled_dataset) == (num_tile_rows * num_tile_cols * len(dataset)), "Incorrect number of tiles"

    def test_tiler_consistency(self, mocker):
        # Test that the tiler and tile transform are consistent
        rng = np.random.default_rng()
        rnd_tile_size = rng.integers(low=100, high=500)
        rnd_tile_overlap = rng.random()
        image_size = rng.integers(low=1000, high=5000)
        np_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        dm_image = Image.from_numpy(np_image)

        mock_model = MagicMock(spec=Model)
        mocker.patch("openvino.model_api.tilers.tiler.Tiler.__init__", return_value=None)
        mocker.patch.multiple(Tiler, __abstractmethods__=set())

        tiler = Tiler(model=mock_model)
        tiler.tile_size = rnd_tile_size
        tiler.tiles_overlap = rnd_tile_overlap

        mocker.patch("otx.core.data.dataset.tile.OTXTileTransform.__init__", return_value=None)
        tile_transform = OTXTileTransform()
        tile_transform._tile_size = (rnd_tile_size, rnd_tile_size)
        tile_transform._overlap = (rnd_tile_overlap, rnd_tile_overlap)

        dm_rois = [xywh_to_x1y1x2y2(*roi) for roi in tile_transform._extract_rois(dm_image)]
        # 0 index in tiler is the full image so we skip it
        assert np.allclose(dm_rois, tiler._tile(np_image)[1:])

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

    def test_ov_det_tile_model(
        self,
        tmp_path: Path,
        fxt_accelerator: str,
        fxt_target_dataset_per_task: dict,
    ):
        tile_recipes = [recipe for recipe in pytest.RECIPE_LIST if "detection" in recipe and "tile" in recipe]
        for tile_recipe in tile_recipes:
            engine = Engine.from_config(
                config_path=tile_recipe,
                data_root=fxt_target_dataset_per_task[OTXTaskType.DETECTION.value.lower()],
                work_dir=tmp_path / OTXTaskType.DETECTION,
                device=fxt_accelerator,
            )
            engine.train(max_epochs=1)
            exported_model_path = engine.export()
            assert exported_model_path.exists()
            engine.test(exported_model_path, accelerator="cpu")

            ov_model = OVDetectionModel(model_name=exported_model_path, num_classes=3)

            assert isinstance(ov_model.model, DetectionTiler), "Model should be an instance of DetectionTiler"
            assert engine.datamodule.config.tile_config.tile_size[0] == ov_model.model.tile_size
            assert engine.datamodule.config.tile_config.overlap == ov_model.model.tiles_overlap

    def test_ov_inst_tile_model(
        self,
        tmp_path: Path,
        fxt_accelerator: str,
        fxt_target_dataset_per_task: dict,
    ):
        # Test that tiler is setup correctly for instance segmentation
        tile_recipes = [
            recipe for recipe in pytest.RECIPE_LIST if "instance_segmentation" in recipe and "tile" in recipe
        ]

        for tile_recipe in tile_recipes:
            engine = Engine.from_config(
                config_path=tile_recipe,
                data_root=fxt_target_dataset_per_task[OTXTaskType.INSTANCE_SEGMENTATION.value.lower()],
                work_dir=tmp_path / OTXTaskType.INSTANCE_SEGMENTATION,
                device=fxt_accelerator,
            )
            engine.train(max_epochs=1)
            exported_model_path = engine.export()
            assert exported_model_path.exists()
            engine.test(exported_model_path, accelerator="cpu")

            ov_model = OVInstanceSegmentationModel(model_name=exported_model_path, num_classes=3)

            assert isinstance(
                ov_model.model,
                InstanceSegmentationTiler,
            ), "Model should be an instance of InstanceSegmentationTiler"
            assert engine.datamodule.config.tile_config.tile_size[0] == ov_model.model.tile_size
            assert engine.datamodule.config.tile_config.overlap == ov_model.model.tiles_overlap
