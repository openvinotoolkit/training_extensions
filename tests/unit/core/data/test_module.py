# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from importlib_resources import files
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from otx.core.config.data import (
    SubsetConfig,
    TileConfig,
    UnlabeledDataConfig,
)
from otx.core.data import module as target_file
from otx.core.data.module import (
    OTXDataModule,
    OTXTaskType,
)

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DmDataset


class TestModule:
    @pytest.fixture()
    def fxt_config(self) -> DictConfig:
        train_subset = MagicMock(spec=SubsetConfig)
        train_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 4}},
        )
        train_subset.num_workers = 0
        train_subset.batch_size = 4
        train_subset.input_size = None
        train_subset.subset_name = "train_1"
        val_subset = MagicMock(spec=SubsetConfig)
        val_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 3}},
        )
        val_subset.num_workers = 0
        val_subset.batch_size = 3
        val_subset.input_size = None
        val_subset.subset_name = "val_1"
        test_subset = MagicMock(spec=SubsetConfig)
        test_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 3}},
        )
        test_subset.num_workers = 0
        test_subset.batch_size = 1
        test_subset.input_size = None
        test_subset.subset_name = "test_1"
        unlabeled_subset = MagicMock(spec=UnlabeledDataConfig)
        unlabeled_subset.data_root = None
        tile_config = MagicMock(spec=TileConfig)
        tile_config.enable_tiler = False

        mock = MagicMock(spec=DictConfig)
        mock.task = "MULTI_LABEL_CLS"
        mock.data_format = "coco_instances"
        mock.data_root = "."
        mock.train_subset = train_subset
        mock.val_subset = val_subset
        mock.test_subset = test_subset
        mock.unlabeled_subset = unlabeled_subset
        mock.tile_config = tile_config

        return mock

    @pytest.fixture()
    def mock_dm_dataset(self, mocker) -> MagicMock:
        return mocker.patch("otx.core.data.module.DmDataset.import_from")

    @pytest.fixture()
    def mock_otx_dataset_factory(self, mocker) -> MagicMock:
        return mocker.patch("otx.core.data.module.OTXDatasetFactory")

    @pytest.fixture()
    def mock_data_filtering(self, mocker) -> MagicMock:
        def func(
            dataset: DmDataset,
            data_format: str,
            unannotated_items_ratio: float,
            ignore_index: int | None,
        ) -> DmDataset:
            del data_format
            del unannotated_items_ratio
            del ignore_index
            return dataset

        return mocker.patch("otx.core.data.module.pre_filtering", side_effect=func)

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
        mock_data_filtering,
        task,
        fxt_config,
    ) -> None:
        # Dataset will have "train_0", "train_1", "val_0", ..., "test_1" subsets
        mock_dm_subsets = {f"{name}_{idx}": MagicMock() for name in ["train", "val", "test"] for idx in range(2)}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets

        module = OTXDataModule(
            task=task,
            data_format=fxt_config.data_format,
            data_root=fxt_config.data_root,
            train_subset=fxt_config.train_subset,
            val_subset=fxt_config.val_subset,
            test_subset=fxt_config.test_subset,
        )

        assert module.train_dataloader().batch_size == 4
        assert module.val_dataloader().batch_size == 3
        assert module.test_dataloader().batch_size == 1
        assert module.predict_dataloader().batch_size == 1
        assert mock_otx_dataset_factory.create.call_count == 3
        assert fxt_config.train_subset.input_size is None
        assert fxt_config.val_subset.input_size is None
        assert fxt_config.test_subset.input_size is None

    def test_init_input_size(
        self,
        mock_dm_dataset,
        mock_otx_dataset_factory,
        mock_data_filtering,
        fxt_config,
    ) -> None:
        # Dataset will have "train_0", "train_1", "val_0", ..., "test_1" subsets
        mock_dm_subsets = {f"{name}_{idx}": MagicMock() for name in ["train", "val", "test"] for idx in range(2)}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets
        fxt_config.train_subset.input_size = None
        fxt_config.val_subset.input_size = None
        fxt_config.test_subset.input_size = (800, 800)

        OTXDataModule(
            task=OTXTaskType.MULTI_CLASS_CLS,
            data_format=fxt_config.data_format,
            data_root=fxt_config.data_root,
            train_subset=fxt_config.train_subset,
            val_subset=fxt_config.val_subset,
            test_subset=fxt_config.test_subset,
            input_size=(1200, 1200),
        )

        assert fxt_config.train_subset.input_size == (1200, 1200)
        assert fxt_config.val_subset.input_size == (1200, 1200)
        assert fxt_config.test_subset.input_size == (800, 800)

    @pytest.fixture()
    def mock_adapt_input_size_to_dataset(self, mocker) -> MagicMock:
        return mocker.patch.object(target_file, "adapt_input_size_to_dataset", return_value=(1234, 1234))

    def test_init_adaptive_input_size(
        self,
        mock_dm_dataset,
        mock_otx_dataset_factory,
        mock_data_filtering,
        fxt_config,
        mock_adapt_input_size_to_dataset,
    ) -> None:
        # Dataset will have "train_0", "train_1", "val_0", ..., "test_1" subsets
        mock_dm_subsets = {f"{name}_{idx}": MagicMock() for name in ["train", "val", "test"] for idx in range(2)}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets
        fxt_config.train_subset.input_size = None
        fxt_config.val_subset.input_size = (1000, 1000)
        fxt_config.test_subset.input_size = None

        OTXDataModule(
            task=OTXTaskType.MULTI_CLASS_CLS,
            data_format=fxt_config.data_format,
            data_root=fxt_config.data_root,
            train_subset=fxt_config.train_subset,
            val_subset=fxt_config.val_subset,
            test_subset=fxt_config.test_subset,
            adaptive_input_size="auto",
        )

        assert fxt_config.train_subset.input_size == (1234, 1234)
        assert fxt_config.val_subset.input_size == (1000, 1000)
        assert fxt_config.test_subset.input_size == (1234, 1234)

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

    def test_hparams_initial_is_loggable(
        self,
        mock_dm_dataset,
        mock_otx_dataset_factory,
        mock_data_filtering,
        fxt_real_tv_cls_config,
        tmpdir,
    ) -> None:
        # Dataset will have "train", "val", and "test" subsets
        mock_dm_subsets = {name: MagicMock() for name in ["train", "val", "test"]}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets
        module = OTXDataModule(**fxt_real_tv_cls_config)
        logger = CSVLogger(tmpdir)
        logger.log_hyperparams(module.hparams_initial)
        logger.save()

        hparams_path = Path(logger.log_dir) / "hparams.yaml"
        assert hparams_path.exists()
