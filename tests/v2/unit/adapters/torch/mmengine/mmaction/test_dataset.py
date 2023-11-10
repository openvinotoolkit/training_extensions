
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.v2.adapters.torch.mmengine.mmaction.dataset import MMActionDataset, get_default_pipeline
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TrainType
from pytest_mock.plugin import MockerFixture


def test_get_default_pipeline() -> None:
    expected_pipeline = [
        {
            "type": "SampleFrames", 
            "clip_len": 8, 
            "frame_interval": 4,
            "num_clips": 1,
        },
        {"type": "mmaction.RawFrameDecode"},
        {"type": "mmaction.Resize", "scale": (-1, 256)},
        {"type": "mmaction.FormatShape", "input_format": "NCTHW"},
        {"type": "mmaction.PackActionInputs"},
    ]
    result = get_default_pipeline()
    assert result == expected_pipeline

class TestDataset:
    def test_init(self) -> None:
        dataset = MMActionDataset()
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

        dataset = MMActionDataset(
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
        mock_set_datumaro_adapters = mocker.patch("otx.v2.adapters.torch.mmengine.mmaction.dataset.Dataset.set_datumaro_adapters")
        dataset = MMActionDataset()
        dataset.train_type = TrainType.Incremental
        mock_label_schema = mocker.MagicMock()
        mock_label_schema.get_groups.return_value = ["test1"]
        mock_label_schema.get_labels.return_value = ["test1"]
        dataset.label_schema = mock_label_schema

        dataset._initialize()
        mock_set_datumaro_adapters.assert_called_once()
        assert dataset.base_dataset.__name__ == "OTXActionClsDataset"

    def test_build_dataset(self, mocker: MockerFixture) -> None:
        mock_initialize = mocker.patch("otx.v2.adapters.torch.mmengine.mmaction.dataset.Dataset._initialize")
        mock_mmpretrain_build_dataset = mocker.patch("mmaction.registry.DATASETS.build")
        mock_mmpretrain_build_dataset.return_value = mocker.MagicMock()

        # Invalid subset
        dataset = MMActionDataset(
            train_data_roots="train/data/roots",
            train_ann_files="train/ann/files",
        )
        with pytest.raises(ValueError, match="invalid is not supported subset"):
            dataset.build_dataset(subset="invalid")
        mock_initialize.assert_called_once()

        mock_label_schema = mocker.MagicMock()
        mock_label_schema.get_labels.return_value = ["label1"]
        dataset.label_schema = mock_label_schema

        # otx_dataset < 1
        mock_dataset_entity = mocker.MagicMock()
        mock_dataset_entity.get_subset.return_value = []
        dataset.dataset_entity = mock_dataset_entity
        result = dataset.build_dataset(subset="train")
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

        result = dataset.build_dataset(subset="train")
        mock_base_dataset.assert_called_once()

        # config is not None
        # config is str
        mock_config = Config({
            "dataset": {}
        })
        mock_fromfile = mocker.patch("otx.v2.adapters.torch.mmengine.mmaction.dataset.Config.fromfile")
        mock_fromfile.return_value = mock_config
        result = dataset.build_dataset(subset="train", config="test.yaml")
        mock_fromfile.assert_called_once_with(filename="test.yaml")
        mock_mmpretrain_build_dataset.assert_called_once()

        # config is dict
        result = dataset.build_dataset(subset="train", config={})
        mock_mmpretrain_build_dataset.assert_called()

        # config is Config with pipeline
        result = dataset.build_dataset(subset="train", config=Config({}), pipeline=[mocker.MagicMock()])
        mock_mmpretrain_build_dataset.assert_called()

    def test_build_dataloader(self, mocker: MockerFixture) -> None:
        # dataset is None
        dataset = MMActionDataset()
        assert dataset.build_dataloader(dataset=None) is None

        mock_get_dist_info = mocker.patch("otx.v2.adapters.torch.mmengine.mmaction.dataset.get_dist_info", return_value=(1, 2))
        mock_torch_dataloader = mocker.patch("otx.v2.adapters.torch.mmengine.mmaction.dataset.TorchDataLoader")
        mock_patial = mocker.patch("otx.v2.adapters.torch.mmengine.mmaction.dataset.partial")
        mock_patial.return_value = mocker.MagicMock()
        mock_dataset = mocker.MagicMock()
        dataset.build_dataloader(
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
            drop_last=True,
            persistent_workers=False,
        )
