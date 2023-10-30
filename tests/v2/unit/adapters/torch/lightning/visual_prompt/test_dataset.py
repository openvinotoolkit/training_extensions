# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.v2.adapters.torch.lightning.visual_prompt.dataset import VisualPromptDataset as Dataset
from otx.v2.api.entities.task_type import TaskType
from pytest_mock.plugin import MockerFixture


class TestVisualPromptDataset:
    def test_init(self) -> None:
        dataset = Dataset()
        assert dataset.train_data_roots is None
        assert dataset.train_ann_files is None
        assert dataset.val_data_roots is None
        assert dataset.val_ann_files is None
        assert dataset.test_data_roots is None
        assert dataset.test_ann_files is None
        assert dataset.unlabeled_data_roots is None
        assert dataset.unlabeled_file_list is None
        assert dataset.task == TaskType.VISUAL_PROMPTING
        assert dataset.train_type is None
        assert dataset.data_format is None
        assert dataset.initialize is False

        dataset = Dataset(
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
        mock_initialize = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.dataset.VisualPromptDataset._initialize")

        # Invalid subset
        dataset = Dataset(
            train_data_roots="train/data/roots",
            train_ann_files="train/ann/files",
        )
        mock_dataset_entity = mocker.MagicMock()
        # Empty Dataset
        mock_dataset_entity.get_subset.return_value = []
        dataset.dataset_entity = mock_dataset_entity
        assert dataset.build_dataset(subset="train") is None
        mock_initialize.assert_called_once_with()

        # Predict Dataset
        dataset.dataset_entity = mock_dataset_entity
        dataset.build_dataset(subset="predict")
        assert dataset.build_dataset(subset="predict") is None

        # Dataset
        mock_vp_dataset = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.dataset.OTXVisualPromptingDataset")
        mock_dataset_entity.get_subset.return_value = mocker.MagicMock()
        mock_dataset_entity.get_subset.return_value.__len__.return_value = 3
        dataset.dataset_entity = mock_dataset_entity
        dataset.build_dataset(subset="train")
        mock_vp_dataset.assert_called_once_with(
            dataset=mock_dataset_entity.get_subset.return_value,
            image_size=1024,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            offset_bbox=20,
            pipeline=None,
        )

    def test_build_dataloader(self, mocker: MockerFixture) -> None:
        mock_build_dataloader = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.dataset.LightningDataset.build_dataloader")
        dataset = Dataset(
            train_data_roots="train/data/roots",
            train_ann_files="train/ann/files",
        )
        mock_torch_dataset = mocker.MagicMock()
        dataset.build_dataloader(
            dataset=mock_torch_dataset,
            batch_size=2,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
        from otx.v2.adapters.torch.lightning.visual_prompt.modules.datasets.pipelines import collate_fn
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
