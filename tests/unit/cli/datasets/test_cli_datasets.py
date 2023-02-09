# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest

# from otx.algorithms.anomaly.adapters.anomalib.data.dataset import (
#     AnomalyClassificationDataset,
#     AnomalyDetectionDataset,
#     AnomalySegmentationDataset,
# )
from otx.algorithms.classification.utils import ClassificationDatasetAdapter
from otx.api.entities.model_template import TaskType
from otx.cli.datasets import get_dataset_class
from otx.cli.datasets.action_classification.dataset import ActionClassificationDataset
from otx.cli.datasets.action_detection.dataset import ActionDetectionDataset
from otx.cli.datasets.image_classification.dataset import ImageClassificationDataset
from otx.cli.datasets.instance_segmentation.dataset import InstanceSegmentationDataset
from otx.cli.datasets.object_detection.dataset import ObjectDetectionDataset
from otx.cli.datasets.rotated_detection.dataset import RotatedDetectionDataset
from otx.cli.datasets.semantic_segmentation.dataset import SemanticSegmentationDataset
from tests.test_suite.e2e_test_system import e2e_pytest_unit

TASKTYPE = {
    # TaskType.ANOMALY_CLASSIFICATION: AnomalyClassificationDataset,
    # TaskType.ANOMALY_DETECTION: AnomalyDetectionDataset,
    # TaskType.ANOMALY_SEGMENTATION: AnomalySegmentationDataset,
    TaskType.CLASSIFICATION: ImageClassificationDataset,
    TaskType.DETECTION: ObjectDetectionDataset,
    TaskType.INSTANCE_SEGMENTATION: InstanceSegmentationDataset,
    TaskType.ROTATED_DETECTION: RotatedDetectionDataset,
    TaskType.SEGMENTATION: SemanticSegmentationDataset,
    TaskType.ACTION_CLASSIFICATION: ActionClassificationDataset,
    TaskType.ACTION_DETECTION: ActionDetectionDataset,
}

mock_function_without_unlabeled = {
    "action_classification": (ActionClassificationDataset, "load_cls_dataset"),
    "action_detection": (ActionDetectionDataset, "load_det_dataset"),
}

mock_function_with_unlabeled = {
    "object_detection": (ObjectDetectionDataset, "load_dataset_items_coco_format"),
    "instance_segmentation": (InstanceSegmentationDataset, "load_dataset_items_coco_format"),
    "rotated_detection": (RotatedDetectionDataset, "load_dataset_items_coco_format"),
    "semantic_segmentation": (SemanticSegmentationDataset, "load_dataset_items"),
}


@pytest.fixture
def mock_train_subset():
    return {"ann_file": "path/to/ann_file_train", "data_root": "path/to/data_root_train"}


@pytest.fixture
def mock_val_subset():
    return {"ann_file": "path/to/ann_file_val", "data_root": "path/to/data_root_val"}


@pytest.fixture
def mock_test_subset():
    return {"ann_file": "path/to/ann_file_test", "data_root": "path/to/data_root_test"}


@pytest.fixture
def mock_unlabeled_subset():
    return {"data_root": "path/to/data_root_unlabeled", "file_list": "path/to/file_list_unlabeled"}


@e2e_pytest_unit
@pytest.mark.parametrize("task_type,expected_results", [(key, value) for key, value in TASKTYPE.items()])
def test_datasets_get_dataset_class(task_type, expected_results):
    """Check get_dataset_clsss from otx.cli.datasets."""
    dataset_class = get_dataset_class(task_type)
    assert dataset_class == expected_results


@e2e_pytest_unit
@pytest.mark.parametrize("task", [TaskType.NULL, "unexpected"])
def test_datasets_get_dataset_class_raise_value_error(task):
    """Check raising Error in get_dataset_clsss from otx.cli.datasets.

    <Steps>
        1. Check raising ValueError with NULL task_type
        2. Check raising ValueError with unexpected task_type
    """
    with pytest.raises(ValueError):
        get_dataset_class(task)


@e2e_pytest_unit
@pytest.mark.parametrize(
    "task_type,target_mock_function", [(key, value) for key, value in mock_function_without_unlabeled.items()]
)
def test_cli_datasets_without_unlabeled(
    task_type, target_mock_function, mocker, mock_train_subset, mock_val_subset, mock_test_subset
):
    """Check dataset init working well (unsupported unlabeled data)."""
    mock_items = [{"data_key_1": "data_1"}, {"data_key_2": "data_2"}]
    target_class, mock_function = target_mock_function
    mocker.patch(f"otx.cli.datasets.{task_type}.dataset.{mock_function}", return_value=mock_items)

    dataset = target_class(mock_train_subset, mock_val_subset, mock_test_subset)

    assert dataset._items == mock_items * 3


@e2e_pytest_unit
@pytest.mark.parametrize(
    "task_type,target_mock_function", [(key, value) for key, value in mock_function_with_unlabeled.items()]
)
def test_cli_datasets_with_unlabeled(
    task_type, target_mock_function, mocker, mock_train_subset, mock_val_subset, mock_test_subset, mock_unlabeled_subset
):
    """Check dataset init working well (supported unlabeled data)."""
    mock_items = [{"data_key_1": "data_1"}, {"data_key_2": "data_2"}]
    target_class, mock_function = target_mock_function
    mocker.patch(f"otx.cli.datasets.{task_type}.dataset.{mock_function}", return_value=mock_items)
    mocker.patch(f"otx.cli.datasets.{task_type}.dataset.load_unlabeled_dataset_items", return_value=mock_items)

    dataset = target_class(mock_train_subset, mock_val_subset, mock_test_subset, mock_unlabeled_subset)

    assert dataset._items == mock_items * 4


@e2e_pytest_unit
def test_ImageClassificationDataset(
    mocker, mock_train_subset, mock_val_subset, mock_test_subset, mock_unlabeled_subset
):
    """Check classification dataset init working well."""
    mock_classification_init = mocker.patch.object(ClassificationDatasetAdapter, "__init__", return_value=None)
    ImageClassificationDataset(
        train_subset=mock_train_subset,
        val_subset=mock_val_subset,
        test_subset=mock_test_subset,
        unlabeled_subset=mock_unlabeled_subset,
    )
    mock_classification_init.assert_called_once_with(
        mock_train_subset.get("ann_file"),
        mock_train_subset.get("data_root"),
        mock_val_subset.get("ann_file"),
        mock_val_subset.get("data_root"),
        mock_test_subset.get("ann_file"),
        mock_test_subset.get("data_root"),
        mock_unlabeled_subset.get("data_root"),
        mock_unlabeled_subset.get("file_list"),
    )
