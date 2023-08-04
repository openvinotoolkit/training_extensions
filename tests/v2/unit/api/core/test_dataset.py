"""OTX V2 API-core Unit-Test codes (Dataset)."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from otx.v2.api.core.dataset import BaseDataset, BaseDatasetAdapter, get_dataset_adapter
from otx.v2.api.entities.task_type import TaskType, TrainType
from pytest_mock.plugin import MockerFixture


def test_get_dataset_adapter(mocker: MockerFixture) -> None:
    """Test function for the get_dataset_adapter function in the otx.v2.api.core.dataset module.

    This function tests the get_dataset_adapter function with two different task types and train types.
    It asserts that the correct dataset adapter is returned for each test case.

    Args:
    ----
        mocker (pytest_mock.MockerFixture): A pytest-mock fixture for mocking objects.

    Returns:
    -------
        None
    """
    mock_getattr = mocker.patch("otx.v2.api.core.dataset.getattr")
    # Test case 1: ANOMALY_CLASSIFICATION task with Incremental train type
    get_dataset_adapter(
        task_type=TaskType.ANOMALY_CLASSIFICATION,
        train_type=TrainType.Incremental,
        train_data_roots="path/to/train/data",
        train_ann_files="path/to/train/annotations",
        val_data_roots="path/to/val/data",
        val_ann_files="path/to/val/annotations",
        test_data_roots="path/to/test/data",
        test_ann_files="path/to/test/annotations",
    )
    assert mock_getattr.call_args[0][-1] == "AnomalyClassificationDatasetAdapter"

    # Test case 2: CLASSIFICATION task with Incremental train type
    get_dataset_adapter(
        task_type=TaskType.CLASSIFICATION,
        train_type=TrainType.Incremental,
        train_data_roots="path/to/train/data",
        train_ann_files="path/to/train/annotations",
        val_data_roots="path/to/val/data",
        val_ann_files="path/to/val/annotations",
        test_data_roots="path/to/test/data",
        test_ann_files="path/to/test/annotations",
        unlabeled_data_roots="path/to/unlabeled/data",
        unlabeled_file_list="path/to/unlabeled/file/list",
    )
    assert mock_getattr.call_args[0][-1] == "ClassificationDatasetAdapter"


class TestBaseDatasetAdapter:
    """This class contains unit tests for the BaseDatasetAdapter class."""

    def test_get_otx_dataset_not_implemented(self) -> None:
        """Test that the get_otx_dataset method raises a NotImplementedError."""
        adapter = BaseDatasetAdapter(TaskType.CLASSIFICATION)
        with pytest.raises(NotImplementedError):
            adapter.get_otx_dataset()

    def test_get_label_schema_not_implemented(self) -> None:
        """Test that the get_label_schema method raises a NotImplementedError."""
        adapter = BaseDatasetAdapter(TaskType.CLASSIFICATION)
        with pytest.raises(NotImplementedError):
            adapter.get_label_schema()


class TestBaseDataset:
    """A test suite for the BaseDataset class."""

    def test_init(self) -> None:
        """Test the initialization of the BaseDataset class.

        Steps:
        1. Create a BaseDataset object with specified parameters.
        2. Assert that the object's attributes match the specified parameters.
        """
        base_dataset = BaseDataset(
            task="classification",
            train_type="Incremental",
            train_data_roots="/path/to/train/data",
            train_ann_files="/path/to/train/annotations",
            val_data_roots="/path/to/val/data",
            val_ann_files="/path/to/val/annotations",
            test_data_roots="/path/to/test/data",
            test_ann_files="/path/to/test/annotations",
            unlabeled_data_roots="/path/to/unlabeled/data",
            unlabeled_file_list="/path/to/unlabeled/file/list",
            data_format="custom",
        )
        assert base_dataset.task == "classification"
        assert base_dataset.train_type == "Incremental"
        assert base_dataset.train_data_roots == "/path/to/train/data"
        assert base_dataset.train_ann_files == "/path/to/train/annotations"
        assert base_dataset.val_data_roots == "/path/to/val/data"
        assert base_dataset.val_ann_files == "/path/to/val/annotations"
        assert base_dataset.test_data_roots == "/path/to/test/data"
        assert base_dataset.test_ann_files == "/path/to/test/annotations"
        assert base_dataset.unlabeled_data_roots == "/path/to/unlabeled/data"
        assert base_dataset.unlabeled_file_list == "/path/to/unlabeled/file/list"
        assert base_dataset.data_format == "custom"

    def test_set_datumaro_adapters(self, mocker: MockerFixture) -> None:
        """Test the set_datumaro_adapters method of the BaseDataset class.

        Steps:
        1. Mock the configure_task_type, configure_train_type, and get_dataset_adapter functions.
        2. Create a BaseDataset object with specified parameters.
        3. Call the set_datumaro_adapters method on the object.
        4. Assert that the object's dataset_adapter, dataset_entity, and label_schema attributes are not None.
        """
        mocker.patch("otx.v2.api.core.dataset.configure_task_type", return_value=("classification", "imagenet"))
        mocker.patch("otx.v2.api.core.dataset.configure_train_type", return_value="Incremental")
        mocker.patch("otx.v2.api.core.dataset.get_dataset_adapter")
        base_dataset = BaseDataset(
            train_data_roots="/path/to/train/data",
        )
        base_dataset.set_datumaro_adapters()
        assert base_dataset.dataset_adapter is not None
        assert base_dataset.dataset_entity is not None
        assert base_dataset.label_schema is not None

    def test_subset_dataloader(self, mocker: MockerFixture) -> None:
        """Test the subset_dataloader method of the BaseDataset class.

        Steps:
        1. Create a BaseDataset object.
        2. Spy on the subset_dataloader method of the object.
        3. Call the subset_dataloader method on the object with the "train" parameter.
        4. Assert that the subset_dataloader method was called once.
        """
        base_dataset = BaseDataset()
        mock_subset_dataloader = mocker.spy(base_dataset, "subset_dataloader")
        base_dataset.subset_dataloader("train")

        mock_subset_dataloader.assert_called()
        assert mock_subset_dataloader.call_count == 1

    def test_num_classes(self) -> None:
        """Test the num_classes method of the BaseDataset class.

        Steps:
        1. Create a BaseDataset object.
        2. Attempt to access the num_classes attribute of the object.
        3. Assert that a NotImplementedError is raised.
        """
        dataset = BaseDataset()
        with pytest.raises(NotImplementedError):
            _ = dataset.num_classes

    def test_train_data_roots(self) -> None:
        """Test the train_data_roots attribute of the BaseDataset class.

        Steps:
        1. Create a BaseDataset object.
        2. Set the train_data_roots attribute of the object to a specified value.
        3. Assert that the train_data_roots attribute of the object matches the specified value.
        """
        dataset = BaseDataset()
        dataset.train_data_roots = "/path/to/train/data"
        assert dataset.train_data_roots == "/path/to/train/data"

    def test_train_ann_files(self) -> None:
        """Test the train_ann_files attribute of the BaseDataset class.

        Steps:
        1. Create a BaseDataset object.
        2. Set the train_ann_files attribute of the object to a specified value.
        3. Assert that the train_ann_files attribute of the object matches the specified value.
        """
        dataset = BaseDataset()
        dataset.train_ann_files = "/path/to/train/annotations"
        assert dataset.train_ann_files == "/path/to/train/annotations"

    def test_val_data_roots(self) -> None:
        """Test the val_data_roots attribute of the BaseDataset class.

        Steps:
        1. Create a BaseDataset object.
        2. Set the val_data_roots attribute of the object to a specified value.
        3. Assert that the val_data_roots attribute of the object matches the specified value.
        """
        dataset = BaseDataset()
        dataset.val_data_roots = "/path/to/val/data"
        assert dataset.val_data_roots == "/path/to/val/data"

    def test_val_ann_files(self) -> None:
        """Test the val_ann_files attribute of the BaseDataset class.

        Steps:
        1. Create a BaseDataset object.
        2. Set the val_ann_files attribute of the object to a specified value.
        3. Assert that the val_ann_files attribute of the object matches the specified value.
        """
        dataset = BaseDataset()
        dataset.val_ann_files = "/path/to/val/annotations"
        assert dataset.val_ann_files == "/path/to/val/annotations"

    def test_test_data_roots(self) -> None:
        """Test the test_data_roots attribute of the BaseDataset class.

        Steps:
        1. Create a BaseDataset object.
        2. Set the test_data_roots attribute of the object to a specified value.
        3. Assert that the test_data_roots attribute of the object matches the specified value.
        """
        dataset = BaseDataset()
        dataset.test_data_roots = "/path/to/test/data"
        assert dataset.test_data_roots == "/path/to/test/data"

    def test_test_ann_files(self) -> None:
        """Test the test_ann_files attribute of the BaseDataset class.

        Steps:
        1. Create a BaseDataset object.
        2. Set the test_ann_files attribute of the object to a specified value.
        3. Assert that the test_ann_files attribute of the object matches the specified value.
        """
        dataset = BaseDataset()
        dataset.test_ann_files = "/path/to/test/annotations"
        assert dataset.test_ann_files == "/path/to/test/annotations"

    def test_unlabeled_data_roots(self) -> None:
        """Test the unlabeled_data_roots attribute of the BaseDataset class.

        Steps:
        1. Create a BaseDataset object.
        2. Set the unlabeled_data_roots attribute of the object to a specified value.
        3. Assert that the unlabeled_data_roots attribute of the object matches the specified value.
        """
        dataset = BaseDataset()
        dataset.unlabeled_data_roots = "/path/to/unlabeled/data"
        assert dataset.unlabeled_data_roots == "/path/to/unlabeled/data"

    def test_unlabeled_file_list(self) -> None:
        """Test the unlabeled_file_list attribute of the BaseDataset class.

        Steps:
        1. Create a BaseDataset object.
        2. Set the unlabeled_file_list attribute of the object to a specified value.
        3. Assert that the unlabeled_file_list attribute of the object matches the specified value.
        """
        dataset = BaseDataset()
        dataset.unlabeled_file_list = "/path/to/unlabeled/file/list"
        assert dataset.unlabeled_file_list == "/path/to/unlabeled/file/list"
