import os

import pytest

from otx.algorithms.common.configs.training_base import TrainType
from otx.api.entities.subset import Subset
from otx.core.data.adapter import get_dataset_adapter
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.data.test_helpers import (
    TASK_NAME_TO_DATA_ROOT,
    TASK_NAME_TO_TASK_TYPE,
)


@e2e_pytest_unit
@pytest.mark.parametrize("task_name", TASK_NAME_TO_TASK_TYPE.keys())
@pytest.mark.parametrize("train_type", [TrainType.INCREMENTAL.value])
def test_get_dataset_adapter_incremental(task_name, train_type):
    root_path = os.getcwd()
    task_type = TASK_NAME_TO_TASK_TYPE[task_name]
    data_root = TASK_NAME_TO_DATA_ROOT[task_name]

    get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        train_data_roots=os.path.join(root_path, data_root["train"]),
        val_data_roots=os.path.join(root_path, data_root["val"]),
    )

    get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        test_data_roots=os.path.join(root_path, data_root["test"]),
    )


@e2e_pytest_unit
@pytest.mark.parametrize("task_name", ["classification"])
@pytest.mark.parametrize("train_type", [TrainType.SELFSUPERVISED.value])
def test_get_dataset_adapter_selfsl_classification(task_name, train_type):
    root_path = os.getcwd()
    task_type = TASK_NAME_TO_TASK_TYPE[task_name]
    data_root = TASK_NAME_TO_DATA_ROOT[task_name]

    get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        train_data_roots=os.path.join(root_path, data_root["train"]),
    )

    get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        test_data_roots=os.path.join(root_path, data_root["test"]),
    )


@e2e_pytest_unit
@pytest.mark.parametrize("task_name", ["segmentation"])
@pytest.mark.parametrize("train_type", [TrainType.SELFSUPERVISED.value])
def test_get_dataset_adapter_selfsl_segmentation(task_name, train_type):
    root_path = os.getcwd()
    task_type = TASK_NAME_TO_TASK_TYPE[task_name]
    data_root = TASK_NAME_TO_DATA_ROOT[task_name]

    get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        train_data_roots=os.path.join(root_path, data_root["train"]),
    )

    with pytest.raises(ValueError):
        get_dataset_adapter(
            task_type=task_type,
            train_type=train_type,
            test_data_roots=os.path.join(root_path, data_root["test"]),
        )


@e2e_pytest_unit
@pytest.mark.parametrize("task_name", ["classification", "detection", "segmentation"])
@pytest.mark.parametrize("train_type", [TrainType.INCREMENTAL.value])
def test_unlabeled_file_list(task_name, train_type):
    root_path = os.getcwd()
    task_type = TASK_NAME_TO_TASK_TYPE[task_name]
    data_root = TASK_NAME_TO_DATA_ROOT[task_name]

    unlabeled_data_roots = "tests/assets/unlabeled_dataset"
    unlabeled_file_list = "tests/assets/unlabeled_dataset/unlabeled_file_list.txt"

    adapter = get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        train_data_roots=os.path.join(root_path, data_root["train"]),
        val_data_roots=os.path.join(root_path, data_root["val"]),
        unlabeled_data_roots=unlabeled_data_roots,
        unlabeled_file_list=unlabeled_file_list,
    )

    assert len(adapter.dataset[Subset.UNLABELED]) == 8
