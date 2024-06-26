# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from datumaro.components.environment import Environment
from importlib_resources import files
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from otx.core.config.data import (
    DataModuleConfig,
    SubsetConfig,
    TileConfig,
    UnlabeledDataConfig,
)
from otx.core.data.module import (
    OTXDataModule,
    OTXTaskType,
)

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DmDataset


def mock_data_filtering(
    dataset: DmDataset,
    data_format: str,
    unannotated_items_ratio: float,
    ignore_index: int | None,
) -> DmDataset:
    del data_format
    del unannotated_items_ratio
    del ignore_index
    return dataset


class TestModule:
    @pytest.fixture()
    def fxt_config(self) -> dict:
        mock = {}
        mock["data_format"] = "coco_instances"
        mock["data_root"] = "."
        mock["mem_cache_size"] = "1GB"
        train_subset = MagicMock(spec=SubsetConfig)
        train_subset.subset_name = "train_1"
        train_subset.num_workers = 0
        train_subset.batch_size = 4
        train_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 4}},
        )
        mock["train_subset"] = train_subset

        val_subset = MagicMock(spec=SubsetConfig)
        val_subset.subset_name = "val_1"
        val_subset.num_workers = 0
        val_subset.batch_size = 3
        val_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 3}},
        )
        mock["val_subset"] = val_subset

        test_subset = MagicMock(spec=SubsetConfig)
        test_subset.subset_name = "test_1"
        test_subset.num_workers = 0
        test_subset.batch_size = 1
        test_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 3}},
        )
        mock["test_subset"] = test_subset

        unlabeled_subset = MagicMock(spec=UnlabeledDataConfig)
        unlabeled_subset.data_root = None
        mock["unlabeled_subset"] = unlabeled_subset

        tile_config = MagicMock(spec=TileConfig)
        tile_config.enable_tiler = False
        mock["tile_config"] = tile_config

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
            OTXTaskType.INSTANCE_SEGMENTATION,
            OTXTaskType.ACTION_CLASSIFICATION,
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
        # Dataset will have "train_0", "train_1", "val_0", ..., "test_1" subsets
        mock_dm_subsets = {f"{name}_{idx}": MagicMock() for name in ["train", "val", "test"] for idx in range(2)}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets

        mocker.patch("otx.core.data.module.pre_filtering", side_effect=mock_data_filtering)

        module = OTXDataModule(task=task, **fxt_config)

        assert module.train_dataloader().batch_size == 4
        assert module.val_dataloader().batch_size == 3
        assert module.test_dataloader().batch_size == 1
        assert module.predict_dataloader().batch_size == 1
        assert mock_otx_dataset_factory.create.call_count == 3

    @pytest.fixture()
    def fxt_real_tv_cls_config(self) -> DictConfig:
        cfg_path = files("otx") / "recipe" / "_base_" / "data" / "torchvision_base.yaml"
        cfg = OmegaConf.load(cfg_path)
        cfg.data_root = "."
        cfg.train_subset.subset_name = "train"
        cfg.train_subset.num_workers = 0
        cfg.val_subset.subset_name = "val"
        cfg.val_subset.num_workers = 0
        cfg.test_subset.subset_name = "test"
        cfg.test_subset.num_workers = 0
        cfg.unlabeled_subset = {}
        cfg.unlabeled_subset.data_root = None
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

        module = OTXDataModule(**fxt_real_tv_cls_config)
        logger = CSVLogger(tmpdir)
        logger.log_hyperparams(module.hparams_initial)
        logger.save()

        hparams_path = Path(logger.log_dir) / "hparams.yaml"
        assert hparams_path.exists()

    @patch("otx.core.data.module.OTXDatasetFactory.create")
    @patch("otx.core.data.module.DmDataset.import_from")
    def test_data_format_check(
        self,
        mock_dm_dataset,
        fxt_config,
        mocker,
        caplog,
    ) -> None:
        print("#######", fxt_config)
        fxt_config["mem_cache_size"] = "0GB"
        fxt_config["tile_config"] = TileConfig(enable_tiler=False)
        print("#######", fxt_config)

        # Dataset will have "train_0", "train_1", "val_0", ..., "test_1" subsets
        mock_dm_subsets = {f"{name}_{idx}": MagicMock() for name in ["train", "val", "test"] for idx in range(2)}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets

        mocker.patch("otx.core.data.module.pre_filtering", side_effect=mock_data_filtering)

        with patch.object(Environment, "detect_dataset", return_value=["voc", "voc_classification"]):
            # with pytest.raises(ValueError, match="Invalid data root:"):
            OTXDataModule(task="MULTI_LABEL_CLS", **fxt_config)

        assert "Invalid data format:" in caplog.text
        assert "Replace data_format:" in caplog.text

        with patch.object(Environment, "detect_dataset", return_value=[]), pytest.raises(
            ValueError,
            match="Invalid data root:",
        ):
            OTXDataModule(task="MULTI_LABEL_CLS", **fxt_config)
