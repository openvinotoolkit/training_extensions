import pytest
import socket
from contextlib import closing
from copy import deepcopy

import torch

from otx.cli.utils.multi_gpu import (
    _get_free_port,
    get_gpu_ids,
    set_arguments_to_argv,
    MultiGPUManager,
)


def test_get_free_port():
    free_port = _get_free_port()

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", free_port))

def test_get_gpu_ids():
    num_available_gpu = torch.cuda.device_count()

    gpus = []
    for i in range(0, num_available_gpu, 2):
        gpus.append(i)

    expected_result = deepcopy(gpus)
    gpus.append(num_available_gpu+2)

    assert get_gpu_ids(",".join([str(val) for val in gpus])) == expected_result

def test_get_gpu_ids_with_wrong_args():
    with pytest.raises(ValueError):
        get_gpu_ids("abcd")

@pytest.fixture
def mock_argv_without_params(mocker):
    mock_sys = mocker.patch("otx.cli.utils.multi_gpu.sys")
    mock_sys.argv = ["--a_key", "a_val", "--b_key"]
    return mock_sys.argv

@pytest.fixture
def mock_argv_with_params(mock_argv_without_params):
    mock_argv_without_params.extend(["params", "--c_key", "c_val", "--d_key"])
    return mock_argv_without_params

def test_set_arguments_to_argv_key_exist(mock_argv_without_params):
    """Test a case where key already exists and value exists."""
    other_val = "other_val"
    set_arguments_to_argv("--a_key", other_val)

    assert mock_argv_without_params[1] == other_val

def test_set_arguments_to_argv_key_exist_none_val(mock_argv_without_params):
    """Test a case where key already exists in argv and value doesn't exists."""
    expected_result = deepcopy(mock_argv_without_params)
    set_arguments_to_argv("--a_key")

    assert mock_argv_without_params == expected_result

def test_set_arguments_to_argv_key(mock_argv_with_params):
    """Test a case where key to set doesn't exists in argv and order of key is before params and vlaue exists."""
    set_arguments_to_argv("--other_key", "other_val")

    param_idx = mock_argv_with_params.index("params")
    new_key_idx = mock_argv_with_params.index("--other_key")

    assert new_key_idx < param_idx
    assert mock_argv_with_params[new_key_idx+1] == "other_val"


def test_set_arguments_to_argv_key_none_val(mock_argv_with_params):
    """Test a case where key to set doesn't exists in argv and order of key is before params and vlaue doesn't exist."""
    set_arguments_to_argv("--other_key")

    param_idx = mock_argv_with_params.index("params")
    new_key_idx = mock_argv_with_params.index("--other_key")

    assert new_key_idx < param_idx
    assert "--other_key" in mock_argv_with_params

def test_set_arguments_to_argv_key_after_param(mock_argv_with_params):
    """Test a case where key to set doesn't exists in argv and order of key is after params and vlaue exists."""
    set_arguments_to_argv("--other_key", "other_val", True)

    param_idx = mock_argv_with_params.index("params")
    new_key_idx = mock_argv_with_params.index("--other_key")

    assert new_key_idx > param_idx
    assert mock_argv_with_params[new_key_idx+1] == "other_val"

def test_set_arguments_to_argv_key_after_param_non_val(mock_argv_with_params):
    """Test a case where key to set doesn't exists in argv and order of key is after params and vlaue doesn't exist."""
    set_arguments_to_argv("--other_key", after_params=True)

    param_idx = mock_argv_with_params.index("params")
    new_key_idx = mock_argv_with_params.index("--other_key")

    assert new_key_idx > param_idx
    assert "--other_key" in mock_argv_with_params


class TestMultiGPUManager:
    def test_init(self, mocker):
        MultiGPUManager(mocker.MagicMock(), "0,1", "localhost:0")
