# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datumaro.components.dataset import Dataset as DmDataset
from importlib_resources import files
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from otx.core.config.data import (
    DataModuleConfig,
    SubsetConfig,
    TileConfig,
)
from otx.core.data.module import (
    OTXDataModule,
    OTXTaskType,
)


def mock_data_filtering(dataset: DmDataset, data_format: str, unannotated_items_ratio: float) -> DmDataset:
    del data_format
    del unannotated_items_ratio
    return dataset


class TestModule:
    @pytest.fixture()
    def fxt_config(self) -> DataModuleConfig:
        mock = MagicMock(spec=DataModuleConfig)
        mock.data_format = "coco_instances"
        mock.data_root = "."
        mock.mem_cache_size = "1GB"
        mock.train_subset = MagicMock(spec=SubsetConfig)
        mock.train_subset.num_workers = 0
        mock.val_subset = MagicMock(spec=SubsetConfig)
        mock.val_subset.num_workers = 0
        mock.test_subset = MagicMock(spec=SubsetConfig)
        mock.test_subset.num_workers = 0
        mock.tile_config = MagicMock(spec=TileConfig)
        mock.tile_config.enable_tiler = False

        return mock

    @patch("otx.core.data.module.OTXDatasetFactory")
    @patch("otx.core.data.module.DmDataset.import_from")
    @pytest.mark.parametrize(
        "task",
        [
            OTXTaskType.MULTI_CLASS_CLS,
            OTXTaskType.MULTI_LABEL_CLS,
            OTXTaskType.H_LABEL_CLS,
            OTXTaskType.DETECTION,
            OTXTaskType.SEMANTIC_SEGMENTATION,
        ],
    )
    def test_init(
        self,
        mock_dm_dataset,
        mock_otx_dataset_factory,
        task,
        fxt_config,
        mocker,
    ) -> None:
        # Our query for subset name for train, val, test
        fxt_config.train_subset.subset_name = "train_1"
        fxt_config.val_subset.subset_name = "val_1"
        fxt_config.test_subset.subset_name = "test_1"

        # Dataset will have "train_0", "train_1", "val_0", ..., "test_1" subsets
        mock_dm_subsets = {f"{name}_{idx}": MagicMock() for name in ["train", "val", "test"] for idx in range(2)}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets

        mocker.patch("otx.core.data.module.pre_filtering", side_effect=mock_data_filtering)

        OTXDataModule(task=task, config=fxt_config)

        assert mock_otx_dataset_factory.create.call_count == 3

    @pytest.fixture()
    def fxt_real_tv_cls_config(self) -> DictConfig:
        cfg_path = files("otx") / "recipe" / "_base_" / "data" / "torchvision_base.yaml"
        cfg = OmegaConf.load(cfg_path)
        cfg = cfg.config
        cfg.data_root = "."
        cfg.train_subset.subset_name = "train"
        cfg.train_subset.num_workers = 0
        cfg.val_subset.subset_name = "val"
        cfg.val_subset.num_workers = 0
        cfg.test_subset.subset_name = "test"
        cfg.test_subset.num_workers = 0
        cfg.mem_cache_size = "1GB"
        cfg.tile_config = {}
        cfg.tile_config.enable_tiler = False
        cfg.auto_num_workers = False
        cfg.device = "auto"
        return cfg

    @patch("otx.core.data.module.OTXDatasetFactory")
    @patch("otx.core.data.module.DmDataset.import_from")
    def test_hparams_initial_is_loggable(
        self,
        mock_dm_dataset,
        mock_otx_dataset_factory,
        fxt_real_tv_cls_config,
        tmpdir,
        mocker,
    ) -> None:
        # Dataset will have "train", "val", and "test" subsets
        mock_dm_subsets = {name: MagicMock() for name in ["train", "val", "test"]}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets

        mocker.patch("otx.core.data.module.pre_filtering", side_effect=mock_data_filtering)

        module = OTXDataModule(task=OTXTaskType.MULTI_CLASS_CLS, config=fxt_real_tv_cls_config)
        logger = CSVLogger(tmpdir)
        logger.log_hyperparams(module.hparams_initial)
        logger.save()

        hparams_path = Path(logger.log_dir) / "hparams.yaml"
        assert hparams_path.exists()
