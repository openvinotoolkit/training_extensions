import os

import pytest

from otx.algorithms.common.configs.training_base import TrainType
from otx.core.data.adapter import get_dataset_adapter
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.data.test_helpers import (
    TASK_NAME_TO_DATA_ROOT,
    TASK_NAME_TO_TASK_TYPE,
)


@e2e_pytest_unit
@pytest.mark.parametrize("task_name", TASK_NAME_TO_TASK_TYPE.keys())
@pytest.mark.parametrize("train_type", [TrainType.Incremental.value])
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
@pytest.mark.parametrize("train_type", [TrainType.Selfsupervised.value])
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
@pytest.mark.parametrize("train_type", [TrainType.Selfsupervised.value])
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
