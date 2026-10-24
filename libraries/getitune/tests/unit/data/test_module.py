# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest
from importlib_resources import files
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms.v2 import Normalize

from getitune.config.data import (
    IntensityConfig,
    SubsetConfig,
    TileConfig,
)
from getitune.data import module as target_file
from getitune.data.module import (
    DataModule,
    DeviceType,
    TaskType,
)


class TestDataModule:
    @pytest.fixture
    def fxt_config(self) -> DictConfig:
        train_subset = MagicMock(spec=SubsetConfig)
        train_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 4}},
        )
        train_subset.num_workers = 0
        train_subset.batch_size = 4
        train_subset.input_size = None
        train_subset.subset_name = "train"
        train_subset.transforms = []
        train_subset.intensity = IntensityConfig()
        val_subset = MagicMock(spec=SubsetConfig)
        val_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 3}},
        )
        val_subset.num_workers = 0
        val_subset.batch_size = 3
        val_subset.input_size = None
        val_subset.subset_name = "val"
        val_subset.intensity = IntensityConfig()
        test_subset = MagicMock(spec=SubsetConfig)
        test_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 3}},
        )
        test_subset.num_workers = 0
        test_subset.batch_size = 1
        test_subset.input_size = None
        test_subset.subset_name = "test"
        test_subset.intensity = IntensityConfig()
        tile_config = MagicMock(spec=TileConfig)
        tile_config.enable_tiler = False

        mock = MagicMock(spec=DictConfig)
        mock.task = "MULTI_LABEL_CLS"
        mock.data_root = "."
        mock.train_subset = train_subset
        mock.val_subset = val_subset
        mock.test_subset = test_subset
        mock.tile_config = tile_config

        return mock

    @pytest.fixture
    def mock_dm_dataset(self, mocker) -> MagicMock:
        return mocker.patch("getitune.data.module.import_dataset")

    @pytest.fixture
    def mock_detect_image_dtype(self, mocker) -> MagicMock:
        return mocker.patch("getitune.data.module.detect_storage_dtype", return_value=("uint8", 3))

    @pytest.fixture
    def mock_dataset_factory(self, mocker) -> MagicMock:
        return mocker.patch("getitune.data.module.DatasetFactory")

    @pytest.mark.parametrize(
        "task",
        [
            TaskType.MULTI_CLASS_CLS,
            TaskType.MULTI_LABEL_CLS,
            TaskType.DETECTION,
            TaskType.SEMANTIC_SEGMENTATION,
            TaskType.INSTANCE_SEGMENTATION,
        ],
    )
    def test_init(
        self,
        mock_dm_dataset,
        mock_dataset_factory,
        mock_detect_image_dtype,
        task,
        fxt_config,
    ) -> None:
        # Make filter_by_subset return a non-empty MagicMock for each subset
        mock_subset = MagicMock()
        mock_subset.__len__ = lambda _: 10
        mock_dm_dataset.return_value.filter_by_subset.return_value = mock_subset
        mock_dm_dataset.return_value.format = "datumaro"

        module = DataModule(
            task=task,
            data_root=fxt_config.data_root,
            train_subset=fxt_config.train_subset,
            val_subset=fxt_config.val_subset,
            test_subset=fxt_config.test_subset,
            input_size=(240, 240),
        )

        assert module.train_dataloader().batch_size == 4
        assert module.val_dataloader().batch_size == 3
        assert module.test_dataloader().batch_size == 1
        assert mock_dataset_factory.create.call_count == 3
        assert fxt_config.train_subset.input_size == (240, 240)
        assert fxt_config.val_subset.input_size == (240, 240)
        assert fxt_config.test_subset.input_size == (240, 240)

    def test_eval_loader_kwargs_uses_recipe_settings(self) -> None:
        """Eval loaders honor the recipe-configured batch size, workers and pinned memory.

        Tiled evaluation memory is bounded by the recipe (the tile recipes set a small
        ``batch_size`` for the val/test subsets), so the DataModule no longer overrides
        the loader settings based on the dataset type — the values come straight from
        the subset config.

        ``pin_memory`` is the exception: it pins into *accelerator* memory, so it is only
        enabled when the datamodule is bound to an accelerator. CPU-only consumers (most
        importantly the OpenVINO evaluation pipeline) must not depend on the GPU runtime.
        """
        from getitune.data.dataset.tile import TileDataset
        from getitune.types.device import DeviceType

        class _FakeTiled(TileDataset):
            def __init__(self):  # bypass heavy TileDataset construction
                pass

            @property
            def collate_fn(self) -> Callable:
                return lambda batch: batch

        dm = DataModule.__new__(DataModule)
        dm.device = DeviceType.gpu
        config = MagicMock(spec=SubsetConfig)
        config.batch_size = 8
        config.num_workers = 4

        plain_dataset = MagicMock()
        plain_dataset.collate_fn = lambda batch: batch
        plain_kwargs = dm._eval_loader_kwargs(plain_dataset, config)
        assert plain_kwargs["batch_size"] == 8
        assert plain_kwargs["num_workers"] == 4
        assert plain_kwargs["pin_memory"] is True

        # Tiled datasets are treated the same way: the recipe controls the settings.
        tiled_kwargs = dm._eval_loader_kwargs(_FakeTiled(), config)
        assert tiled_kwargs["batch_size"] == 8
        assert tiled_kwargs["num_workers"] == 4
        assert tiled_kwargs["pin_memory"] is True

        # A CPU-bound datamodule must not pin batches into accelerator memory.
        dm.device = DeviceType.cpu
        assert dm._eval_loader_kwargs(plain_dataset, config)["pin_memory"] is False

    def test_init_input_size(
        self,
        mock_dm_dataset,
        mock_dataset_factory,
        mock_detect_image_dtype,
        fxt_config,
    ) -> None:
        mock_subset = MagicMock()
        mock_subset.__len__ = lambda _: 10
        mock_dm_dataset.return_value.filter_by_subset.return_value = mock_subset
        mock_dm_dataset.return_value.format = "datumaro"
        fxt_config.train_subset.input_size = None
        fxt_config.val_subset.input_size = None
        fxt_config.test_subset.input_size = (800, 800)

        DataModule(
            task=TaskType.MULTI_CLASS_CLS,
            data_root=fxt_config.data_root,
            train_subset=fxt_config.train_subset,
            val_subset=fxt_config.val_subset,
            test_subset=fxt_config.test_subset,
            input_size=(1200, 1200),
        )

        assert fxt_config.train_subset.input_size == (1200, 1200)
        assert fxt_config.val_subset.input_size == (1200, 1200)
        assert fxt_config.test_subset.input_size == (800, 800)

    @pytest.fixture
    def mock_adapt_input_size_to_dataset(self, mocker) -> MagicMock:
        return mocker.patch.object(target_file, "adapt_input_size_to_dataset", return_value=(1234, 1234))

    @pytest.fixture
    def fxt_real_tv_cls_config(self) -> DictConfig:
        cfg_path = files("getitune") / "recipe" / "_base_" / "data" / "classification.yaml"
        cfg = OmegaConf.load(cfg_path)
        assert isinstance(cfg, DictConfig)
        OmegaConf.set_struct(cfg, False)
        del cfg["input_size"]
        cfg.data_root = "."
        cfg.train_subset.subset_name = "train"
        cfg.train_subset.num_workers = 0
        cfg.train_subset.input_size = None
        cfg.train_subset.augmentations_cpu = []
        cfg.train_subset.augmentations_gpu = []
        cfg.train_subset.intensity = {"storage_dtype": "uint8"}
        cfg.val_subset.subset_name = "val"
        cfg.val_subset.num_workers = 0
        cfg.val_subset.input_size = None
        cfg.val_subset.augmentations_cpu = []
        cfg.val_subset.intensity = {"storage_dtype": "uint8"}
        cfg.test_subset.subset_name = "test"
        cfg.test_subset.num_workers = 0
        cfg.test_subset.input_size = None
        cfg.test_subset.augmentations_cpu = []
        cfg.test_subset.intensity = {"storage_dtype": "uint8"}
        cfg.tile_config = {}
        cfg.tile_config.enable_tiler = False
        cfg.auto_num_workers = False
        cfg.device = "auto"
        return cfg

    def test_hparams_initial_is_loggable(
        self,
        mock_dm_dataset,
        mock_dataset_factory,
        mock_detect_image_dtype,
        fxt_real_tv_cls_config,
        tmpdir,
    ) -> None:
        # Dataset will have "train", "val", and "test" subsets
        mock_subset = MagicMock()
        mock_subset.__len__ = lambda _: 10
        mock_dm_dataset.return_value.filter_by_subset.return_value = mock_subset
        mock_dm_dataset.return_value.format = "datumaro"
        module = DataModule(**fxt_real_tv_cls_config, input_size=(240, 240))
        logger = CSVLogger(tmpdir)
        logger.log_hyperparams(module.hparams_initial)
        logger.save()

        hparams_path = Path(logger.log_dir) / "hparams.yaml"
        assert hparams_path.exists()

    # Fixtures for from_vision_datasets tests
    @pytest.fixture
    def fxt_mock_subset_configs(self) -> dict[str, MagicMock]:
        """Create mock SubsetConfig instances for testing."""
        mock_config = MagicMock(spec=SubsetConfig)
        mock_config.batch_size = 32
        mock_config.num_workers = 2
        mock_config.sampler = DictConfig({"class_path": "torch.utils.data.RandomSampler"})
        mock_config.transforms = []
        mock_config.input_size = (224, 224)
        mock_config.intensity = IntensityConfig()

        configs = {
            "train_subset": deepcopy(mock_config),
            "val_subset": deepcopy(mock_config),
            "test_subset": deepcopy(mock_config),
        }
        configs["train_subset"].subset_name = "train"
        configs["val_subset"].subset_name = "val"
        configs["test_subset"].subset_name = "test"
        return configs

    @pytest.fixture
    def fxt_mock_dataset(self) -> callable:
        """Factory fixture to create mock datasets with specified parameters."""

        def _create_mock_dataset(
            task: TaskType = TaskType.MULTI_CLASS_CLS,
            img_shape: tuple[int, int] = (224, 224),
            transforms: list | None = None,
            label_info: MagicMock | None = None,
        ) -> MagicMock:
            mock_dataset = MagicMock()
            mock_dataset.label_info = label_info or MagicMock()
            mock_dataset.task_type = task
            mock_dataset.image_color_channel = "RGB"
            mock_dataset.transforms = transforms or []
            mock_dataset_item = MagicMock(
                image=MagicMock(shape=(3, *img_shape)),
                img_info=MagicMock(img_shape=img_shape),
            )
            mock_dataset.__iter__ = lambda _: iter([mock_dataset_item])
            return mock_dataset

        return _create_mock_dataset

    def test_from_vision_datasets_basic(
        self, mocker, fxt_mock_subset_configs, fxt_mock_dataset, mock_detect_image_dtype
    ) -> None:
        """Test from_vision_datasets with minimal configuration."""
        # Create mock datasets with shared label_info
        shared_label_info = MagicMock()
        mock_train = fxt_mock_dataset(label_info=shared_label_info)
        mock_val = fxt_mock_dataset(label_info=shared_label_info)
        mock_test = fxt_mock_dataset(label_info=shared_label_info)

        mocker.patch.object(
            DataModule,
            "get_default_subset_configs",
            return_value=fxt_mock_subset_configs,
        )

        # from_vision_datasets requires train_subset with input_size
        train_subset = fxt_mock_subset_configs["train_subset"]

        # Create module from datasets
        module = DataModule.from_vision_datasets(
            train_dataset=mock_train,
            val_dataset=mock_val,
            test_dataset=mock_test,
            train_subset=train_subset,
        )

        # Assertions
        assert module.subsets["train"] == mock_train
        assert module.subsets["val"] == mock_val
        assert module.subsets["test"] == mock_test
        assert module.label_info == shared_label_info
        assert module.task == TaskType.MULTI_CLASS_CLS
        assert module.input_size == (224, 224)

    def test_from_vision_datasets_with_custom_configs(
        self, mocker, fxt_mock_subset_configs, fxt_mock_dataset, mock_detect_image_dtype
    ) -> None:
        """Test from_vision_datasets with custom subset configurations."""
        # Create mock datasets
        shared_label_info = MagicMock()
        mock_train = fxt_mock_dataset(
            task=TaskType.DETECTION,
            img_shape=(640, 640),
            label_info=shared_label_info,
        )
        mock_val = fxt_mock_dataset(
            task=TaskType.DETECTION,
            img_shape=(640, 640),
            label_info=shared_label_info,
        )

        # Create custom configs with explicit input_size (as the Geti application does)
        train_config = MagicMock(spec=SubsetConfig)
        train_config.batch_size = 16
        train_config.num_workers = 4
        train_config.sampler = DictConfig({"class_path": "torch.utils.data.RandomSampler"})
        train_config.transforms = []
        train_config.input_size = (640, 640)
        train_config.intensity = IntensityConfig()

        val_config = MagicMock(spec=SubsetConfig)
        val_config.batch_size = 8
        val_config.num_workers = 2
        val_config.sampler = DictConfig({"class_path": "torch.utils.data.RandomSampler"})
        val_config.transforms = []
        val_config.input_size = (640, 640)
        val_config.intensity = IntensityConfig()

        mocker.patch.object(
            DataModule,
            "get_default_subset_configs",
            return_value=fxt_mock_subset_configs,
        )

        # Create module with custom configs
        module = DataModule.from_vision_datasets(
            train_dataset=mock_train,
            val_dataset=mock_val,
            train_subset=train_config,
            val_subset=val_config,
        )

        # Assertions
        assert module.train_subset == train_config
        assert module.val_subset == val_config
        assert module.test_subset == fxt_mock_subset_configs["test_subset"]
        assert module.task == TaskType.DETECTION
        # input_size should come from train_config, not inferred from image data
        assert module.input_size == (640, 640)

    def test_from_vision_datasets_without_test(
        self, mocker, fxt_mock_subset_configs, fxt_mock_dataset, mock_detect_image_dtype
    ) -> None:
        """Test from_vision_datasets when test_dataset is None (uses val as test)."""
        # Create mock datasets
        shared_label_info = MagicMock()
        mock_train = fxt_mock_dataset(
            task=TaskType.SEMANTIC_SEGMENTATION,
            img_shape=(512, 512),
            label_info=shared_label_info,
        )
        mock_val = fxt_mock_dataset(
            task=TaskType.SEMANTIC_SEGMENTATION,
            img_shape=(512, 512),
            label_info=shared_label_info,
        )

        train_subset = fxt_mock_subset_configs["train_subset"]
        train_subset.input_size = (512, 512)

        mocker.patch.object(
            DataModule,
            "get_default_subset_configs",
            return_value=fxt_mock_subset_configs,
        )

        # Create module without test dataset
        module = DataModule.from_vision_datasets(
            train_dataset=mock_train,
            val_dataset=mock_val,
            test_dataset=None,  # Explicitly None
            train_subset=train_subset,
        )

        # Assertions - test should use val dataset
        assert module.subsets["train"] == mock_train
        assert module.subsets["val"] == mock_val
        assert module.subsets["test"] == mock_val  # Should be val dataset
        assert module.task == TaskType.SEMANTIC_SEGMENTATION
        assert module.input_size == (512, 512)

    def test_from_vision_datasets_label_info_mismatch(self, fxt_mock_subset_configs, fxt_mock_dataset) -> None:
        """Test from_vision_datasets raises error when label_info doesn't match."""
        # Create mock datasets with mismatched label_info
        mock_train = fxt_mock_dataset(label_info=MagicMock())
        mock_val = fxt_mock_dataset(label_info=MagicMock())  # Different label_info

        # Should raise ValueError due to label_info mismatch
        with pytest.raises(ValueError, match="All data meta infos of provided datasets should be the same"):
            DataModule.from_vision_datasets(
                train_dataset=mock_train,
                val_dataset=mock_val,
                train_subset=fxt_mock_subset_configs["train_subset"],
            )

    def test_from_vision_datasets_with_auto_num_workers(
        self, mocker, fxt_mock_subset_configs, fxt_mock_dataset, mock_detect_image_dtype
    ) -> None:
        """Test from_vision_datasets with auto_num_workers enabled."""
        # Create mock datasets
        shared_label_info = MagicMock()
        mock_train = fxt_mock_dataset(
            task=TaskType.DETECTION,
            img_shape=(640, 640),
            label_info=shared_label_info,
        )
        mock_val = fxt_mock_dataset(
            task=TaskType.DETECTION,
            img_shape=(640, 640),
            label_info=shared_label_info,
        )

        train_subset = fxt_mock_subset_configs["train_subset"]
        train_subset.input_size = (640, 640)

        mocker.patch.object(
            DataModule,
            "get_default_subset_configs",
            return_value=fxt_mock_subset_configs,
        )

        # Create module with auto_num_workers
        module = DataModule.from_vision_datasets(
            train_dataset=mock_train,
            val_dataset=mock_val,
            auto_num_workers=True,
            train_subset=train_subset,
        )

        # Assertions
        assert module.auto_num_workers is True
        assert module.device == DeviceType.auto  # Default value
        assert module.input_size == (640, 640)

    @pytest.mark.parametrize(
        ("transforms_source", "expected"),
        [
            (None, (None, None)),
            (
                [Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])],
                ((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
            ),
        ],
        ids=["no normalize", "with normalize"],
    )
    def test_extract_normalization_params(self, transforms_source, expected) -> None:
        """Test CPUAugmentationPipeline._extract_normalization_params."""
        from getitune.data.augmentation.pipeline import CPUAugmentationPipeline

        pipeline = CPUAugmentationPipeline(augmentations=transforms_source)
        result = (pipeline.mean, pipeline.std)
        assert result == expected

    def test_from_vision_datasets_respects_storage_dtype(self, mocker, fxt_mock_subset_configs, tmp_path) -> None:
        """Test that from_vision_datasets preserves caller-provided storage_dtype."""
        # Create mock datasets
        shared_label_info = MagicMock()
        mock_train = MagicMock()
        mock_train.label_info = shared_label_info
        mock_train.task_type = TaskType.MULTI_CLASS_CLS

        mock_val = MagicMock()
        mock_val.label_info = shared_label_info
        mock_val.task_type = TaskType.MULTI_CLASS_CLS

        # Create SubsetConfig with storage_dtype pre-set by caller (as if detect_storage_dtype was called)
        train_subset = SubsetConfig(
            batch_size=4,
            subset_name="train",
            input_size=(224, 224),
            intensity=IntensityConfig(storage_dtype="uint16"),
        )

        mocker.patch.object(
            DataModule,
            "get_default_subset_configs",
            return_value=fxt_mock_subset_configs,
        )

        # Create module from datasets — it should preserve the caller's storage_dtype
        module = DataModule.from_vision_datasets(
            train_dataset=mock_train,
            val_dataset=mock_val,
            train_subset=train_subset,
        )

        # Assert storage_dtype was preserved (not overwritten to uint8)
        assert train_subset.intensity.storage_dtype == "uint16"
        assert module.input_intensity_config is not None
