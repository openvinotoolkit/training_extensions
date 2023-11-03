# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.v2.adapters.torch.mmengine.mmpretrain.dataset import MMPretrainDataset, get_default_pipeline
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TrainType
from pytest_mock.plugin import MockerFixture


def test_get_default_pipeline() -> None:
    expected_pipeline = [
        {"type": "Resize", "scale": [224, 224]},
        {"type": "mmpretrain.PackInputs"},
    ]
    result = get_default_pipeline()
    assert result == expected_pipeline

    expected_pipeline = {
        "train": expected_pipeline,
        "unlabeled": [
            {"type": "Resize", "scale": [224, 224]},
            {"type": "PostAug", "keys": {"img_strong": [
                {"type": "OTXRandAugment", "num_aug": 8, "magnitude": 10},
            ]}},
            {"type": "mmpretrain.PackMultiKeyInputs", "input_key": "img", "multi_key": ["img_strong"]},
        ],
    }
    result = get_default_pipeline(semisl=True)
    assert result == expected_pipeline


class TestDataset:
    def test_init(self) -> None:
        dataset = MMPretrainDataset()
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

        dataset = MMPretrainDataset(
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
        mock_set_datumaro_adapters = mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.dataset.MMPretrainDataset.set_datumaro_adapters")
        dataset = MMPretrainDataset()
        dataset.train_type = TrainType.Incremental
        mock_label_schema = mocker.MagicMock()
        mock_label_schema.get_groups.return_value = ["test1"]
        mock_label_schema.get_labels.return_value = ["test1"]
        dataset.label_schema = mock_label_schema

        dataset._initialize()
        mock_set_datumaro_adapters.assert_called_once()
        mock_label_schema.get_groups.assert_called_once_with(False)
        mock_label_schema.get_labels.assert_called_once_with(include_empty=False)
        assert dataset.base_dataset.__name__ == "OTXClsDataset"

        mock_label_schema = mocker.MagicMock()
        mock_label_schema.get_groups.return_value = ["test1", "test2"]
        mock_label_schema.get_labels.return_value = ["test1", "test2"]
        dataset.label_schema = mock_label_schema
        dataset._initialize()
        mock_label_schema.get_groups.assert_called_once_with(False)
        mock_label_schema.get_labels.assert_called_once_with(include_empty=False)
        assert dataset.base_dataset.__name__ == "OTXMultilabelClsDataset"

        mock_label_schema = mocker.MagicMock()
        mock_label_schema.get_groups.return_value = ["test1", "test2"]
        mock_label_schema.get_labels.return_value = ["test1", "test2", "test3"]
        dataset.label_schema = mock_label_schema
        dataset._initialize()
        mock_label_schema.get_groups.assert_called_once_with(False)
        mock_label_schema.get_labels.assert_called_once_with(include_empty=False)
        assert dataset.base_dataset.__name__ == "OTXHierarchicalClsDataset"

        dataset.train_type = TrainType.Selfsupervised
        dataset._initialize()
        assert dataset.base_dataset.__name__ == "SelfSLDataset"

    def test_build_dataset(self, mocker: MockerFixture) -> None:
        mock_initialize = mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.dataset.MMPretrainDataset._initialize")
        mock_registry = mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.dataset.MMPretrainRegistry")
        mock_registry.return_value.get.return_value = mocker.MagicMock()

        # Invalid subset
        dataset = MMPretrainDataset(
            train_data_roots="train/data/roots",
            train_ann_files="train/ann/files",
        )
        with pytest.raises(ValueError, match="invalid is not supported subset"):
            dataset._build_dataset(subset="invalid")
        mock_initialize.assert_called_once()

        mock_label_schema = mocker.MagicMock()
        mock_label_schema.get_labels.return_value = ["label1"]
        dataset.label_schema = mock_label_schema

        # otx_dataset < 1
        mock_dataset_entity = mocker.MagicMock()
        mock_dataset_entity.get_subset.return_value = []
        dataset.dataset_entity = mock_dataset_entity
        result = dataset._build_dataset(subset="train")
        assert result is None
        mock_dataset_entity.get_subset.assert_called_once_with(Subset.TRAINING)
        mock_label_schema.get_labels.assert_called_once_with(include_empty=False)

        # config is None
        mock_dataset_entity = mocker.MagicMock()
        mock_dataset_entity.get_subset.return_value = ["data1"]
        dataset.dataset_entity = mock_dataset_entity
        mock_base_dataset = mocker.MagicMock()
        mock_base_dataset.__qualname__ = "TestDataset"
        mock_base_dataset.__name__ = "TestDataset"
        mock_base_dataset.return_value = mocker.MagicMock()
        dataset.base_dataset = mock_base_dataset

        result = dataset._build_dataset(subset="train")
        mock_base_dataset.assert_called_once()

        # config is dict
        result = dataset._build_dataset(subset="train", config={})
        mock_registry.return_value.get.return_value.build.assert_called()

        # config is Config with pipeline
        result = dataset._build_dataset(subset="train", config={}, pipeline=[mocker.MagicMock()])
        mock_registry.return_value.get.return_value.build.assert_called()

    def test_build_dataloader(self, mocker: MockerFixture) -> None:
        # dataset is None
        dataset = MMPretrainDataset()
        assert dataset._build_dataloader(dataset=None) is None

        mock_get_dist_info = mocker.patch("otx.v2.adapters.torch.mmengine.dataset.get_dist_info", return_value=(1, 2))
        mock_torch_dataloader = mocker.patch("otx.v2.adapters.torch.dataset.TorchDataLoader")
        mock_patial = mocker.patch("otx.v2.adapters.torch.mmengine.dataset.partial")
        mock_patial.return_value = mocker.MagicMock()
        mock_dataset = mocker.MagicMock()
        dataset._build_dataloader(
            dataset=mock_dataset,
            batch_size=4,
            sampler={},
            num_workers=2,
        )
        mock_get_dist_info.assert_called_once()
        mock_torch_dataloader.assert_called_once_with(
            mock_dataset,
            batch_size=4,
            sampler={},
            num_workers=2,
            collate_fn=mock_patial.return_value,
            pin_memory=False,
            shuffle=False,
            worker_init_fn=None,
            drop_last=False,
            persistent_workers=False,
        )
