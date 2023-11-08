# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig
from otx.v2.adapters.torch.lightning.dataset import LightningDataset
from pytest_mock.plugin import MockerFixture
from otx.v2.api.entities.task_type import TaskType

class TestLightningDataset:
    def test_init(self) -> None:
        dataset = LightningDataset()
        assert dataset.train_data_roots is None
        assert dataset.train_ann_files is None
        assert dataset.val_data_roots is None
        assert dataset.val_ann_files is None
        assert dataset.test_data_roots is None
        assert dataset.test_ann_files is None
        assert dataset.unlabeled_data_roots is None
        assert dataset.unlabeled_file_list is None
        assert dataset.task is None
        assert dataset.train_type is None
        assert dataset.data_format is None
        assert dataset.initialize is False

        dataset = LightningDataset(
            task="Classification",
            train_type="Incremental",
            train_data_roots="train/data/roots",
            train_ann_files="train/ann/files",
            val_data_roots="val/data/roots",
            val_ann_files="val/ann/files",
            test_data_roots="test/data/roots",
            test_ann_files="test/ann/files",
            unlabeled_data_roots="unlabeled/data/roots",
            unlabeled_file_list="unlabeled/files",
        )
        assert dataset.task == "Classification"
        assert dataset.train_type == "Incremental"
        assert dataset.train_data_roots == "train/data/roots"
        assert dataset.train_ann_files=="train/ann/files"
        assert dataset.val_data_roots=="val/data/roots"
        assert dataset.val_ann_files=="val/ann/files"
        assert dataset.test_data_roots=="test/data/roots"
        assert dataset.test_ann_files=="test/ann/files"
        assert dataset.unlabeled_data_roots=="unlabeled/data/roots"
        assert dataset.unlabeled_file_list=="unlabeled/files"

    def test__initialize(self, mocker: MockerFixture) -> None:
        mock_set_datumaro_adapters = mocker.patch("otx.v2.adapters.torch.lightning.dataset.LightningDataset.set_datumaro_adapters")
        dataset = LightningDataset()

        dataset._initialize()
        mock_set_datumaro_adapters.assert_called_once()
        assert dataset.initialize

    def test_build_dataset_visual_prompting(self, mocker: MockerFixture) -> None:
        mock_initialize = mocker.patch("otx.v2.adapters.torch.lightning.dataset.LightningDataset._initialize")

        # Invalid subset
        dataset = LightningDataset(
            train_data_roots="train/data/roots",
            train_ann_files="train/ann/files",
            task="visual_prompting",
        )
        mock_dataset_entity = mocker.MagicMock()
        # Empty Dataset
        mock_dataset_entity.get_subset.return_value = []
        dataset.dataset_entity = mock_dataset_entity
        assert dataset._build_dataset(subset="train") is None
        mock_initialize.assert_called_once_with()

        # Predict Dataset
        dataset.dataset_entity = mock_dataset_entity
        dataset._build_dataset(subset="predict")
        assert dataset._build_dataset(subset="predict") is None

    def test_build_dataloader(self, mocker: MockerFixture) -> None:
        # dataset is None
        dataset = LightningDataset()
        assert dataset._build_dataloader(dataset=None) is None

        mock_torch_dataloader = mocker.patch("otx.v2.adapters.torch.lightning.dataset.BaseTorchDataset._build_dataloader")
        mock_dataset = mocker.MagicMock()
        dataset._build_dataloader(
            dataset=mock_dataset,
            batch_size=4,
            sampler={},
            num_workers=2,
        )
        from otx.v2.adapters.torch.lightning.modules.datasets.pipelines import collate_fn
        mock_torch_dataloader.assert_called_once_with(
            mock_dataset,
            batch_size=4,
            sampler={},
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
        )

    def test_subset_dataloader(self, mocker: MockerFixture) -> None:
        mocker.patch("otx.v2.adapters.torch.dataset.Path.open")
        mocker.patch("otx.v2.adapters.torch.dataset.yaml.safe_load", return_value={"batch_size": 3, "num_workers": 2})
        mocker.patch("otx.v2.adapters.torch.dataset.set_tuple_constructor")

        mock_build_dataset = mocker.patch("otx.v2.adapters.torch.lightning.dataset.LightningDataset._build_dataset")
        mock_build_dataloader = mocker.patch("otx.v2.adapters.torch.lightning.dataset.LightningDataset._build_dataloader")

        dataset = LightningDataset()
        dataset.subset_dataloader(
            subset="train",
            config="test.yaml",
        )
        mock_build_dataset.assert_called_once_with(
            subset="train",
            pipeline=None,
            config={"batch_size": 3, "num_workers": 2},
        )
        mock_build_dataloader.assert_called_once_with(
            dataset=mock_build_dataset.return_value,
            batch_size=3,
            num_workers=2,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
            sampler=None,
            persistent_workers=False,
        )

        mock_config = DictConfig({"key1": "value1"})
        dataset.subset_dataloader(
            subset="test",
            config=mock_config,
        )
        mock_build_dataset.assert_called_with(
            subset="test",
            pipeline=None,
            config=mock_config,
        )

    def test_num_classes(self, mocker: MockerFixture) -> None:
        mock_set_datumaro_adapters = mocker.patch("otx.v2.adapters.torch.lightning.dataset.LightningDataset.set_datumaro_adapters")
        dataset = LightningDataset()
        dataset.label_schema = mocker.Mock()
        dataset.label_schema.get_labels.return_value = ["test1", "test2"]

        assert dataset.num_classes == 2
        assert dataset.initialize
        mock_set_datumaro_adapters.assert_called_once_with()


class TestVisualPromptDataset:
    def test_init(self) -> None:
        dataset = LightningDataset()
        assert dataset.train_data_roots is None
        assert dataset.train_ann_files is None
        assert dataset.val_data_roots is None
        assert dataset.val_ann_files is None
        assert dataset.test_data_roots is None
        assert dataset.test_ann_files is None
        assert dataset.unlabeled_data_roots is None
        assert dataset.unlabeled_file_list is None
        assert dataset.task is None
        assert dataset.train_type is None
        assert dataset.data_format is None
        assert dataset.initialize is False

        dataset = LightningDataset(
            task=TaskType.VISUAL_PROMPTING,
            train_type="Incremental",
            train_data_roots="train/data/roots",
            train_ann_files="train/ann/files",
            val_data_roots="val/data/roots",
            val_ann_files="val/ann/files",
            test_data_roots="test/data/roots",
            test_ann_files="test/ann/files",
            unlabeled_data_roots="unlabeled/data/roots",
            unlabeled_file_list="unlabeled/files",
        )
        assert dataset.task is TaskType.VISUAL_PROMPTING
        assert dataset.train_type == "Incremental"
        assert dataset.train_data_roots == "train/data/roots"
        assert dataset.train_ann_files=="train/ann/files"
        assert dataset.val_data_roots=="val/data/roots"
        assert dataset.val_ann_files=="val/ann/files"
        assert dataset.test_data_roots=="test/data/roots"
        assert dataset.test_ann_files=="test/ann/files"
        assert dataset.unlabeled_data_roots=="unlabeled/data/roots"
        assert dataset.unlabeled_file_list=="unlabeled/files"

    def test_build_dataset(self, mocker: MockerFixture) -> None:
        mock_initialize = mocker.patch("otx.v2.adapters.torch.lightning.dataset.LightningDataset._initialize")

        # Invalid subset
        dataset = LightningDataset(
            train_data_roots="train/data/roots",
            train_ann_files="train/ann/files",
            task="visual_prompting",
        )
        mock_dataset_entity = mocker.MagicMock()
        # Empty Dataset
        mock_dataset_entity.get.return_value = []
        dataset.dataset_entity = mock_dataset_entity
        assert dataset._build_dataset(subset="train") is None
        mock_initialize.assert_called_once_with()

        # Predict Dataset
        dataset.dataset_entity = mock_dataset_entity
        dataset._build_dataset(subset="predict")
        assert dataset._build_dataset(subset="predict") is None

        # Dataset
        mock_vp_dataset = mocker.patch("otx.v2.adapters.torch.lightning.dataset.OTXVisualPromptingDataset")
        mock_dataset_entity.get.return_value = mocker.MagicMock()
        mock_dataset_entity.get.return_value.__len__.return_value = 3
        dataset.dataset_entity = mock_dataset_entity
        dataset._build_dataset(subset="train")
        mock_vp_dataset.assert_called_once_with(
            dataset=mock_dataset_entity.get.return_value,
            image_size=1024,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            offset_bbox=20,
            pipeline=None,
        )

    def test_build_dataloader(self, mocker: MockerFixture) -> None:
        mock_build_dataloader = mocker.patch("otx.v2.adapters.torch.lightning.dataset.LightningDataset._build_dataloader")
        dataset = LightningDataset(
            train_data_roots="train/data/roots",
            train_ann_files="train/ann/files",
            task="visual_prompting",
        )
        mock_torch_dataset = mocker.MagicMock()
        dataset._build_dataloader(
            dataset=mock_torch_dataset,
            batch_size=2,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
        from otx.v2.adapters.torch.lightning.modules.datasets.pipelines import collate_fn
        mock_build_dataloader(
            mock_torch_dataset,
            batch_size=2,
            sampler=None,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            persistent_workers=False,
        )
