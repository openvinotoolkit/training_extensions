import pytest
import os
from otx.core.data.adapter import get_dataset_adapter
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.data.test_helpers import (
    TASK_NAME_TO_TASK_TYPE,
    TASK_NAME_TO_DATA_ROOT
)

@e2e_pytest_unit
@pytest.mark.parametrize("task_name", TASK_NAME_TO_TASK_TYPE.keys())
def test_get_dataset_adapter(task_name):
    root_path = os.getcwd()
    task_type = TASK_NAME_TO_TASK_TYPE[task_name]
    data_root = TASK_NAME_TO_DATA_ROOT[task_name]
    
    get_dataset_adapter(
        task_type = task_type,
        train_data_roots = os.path.join(root_path, data_root["train"]),
        val_data_roots = os.path.join(root_path, data_root["val"]),
        test_data_roots = os.path.join(root_path, data_root["test"]),
    )    