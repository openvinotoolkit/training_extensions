import pytest

from otx.algorithms.common.adapters.mmcv.utils import config_utils
from otx.algorithms.common.adapters.mmcv.utils.config_utils import get_adaptive_num_workers, InputSizeManager


def test_get_adaptive_num_workers(mocker):
    num_gpu = 5
    mock_torch = mocker.patch.object(config_utils, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mock_multiprocessing = mocker.patch.object(config_utils, "multiprocessing")
    mock_multiprocessing.cpu_count.return_value = num_cpu

    assert get_adaptive_num_workers() == num_cpu // num_gpu


def test_get_adaptive_num_workers_no_gpu(mocker):
    num_gpu = 0
    mock_torch = mocker.patch.object(config_utils, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mock_multiprocessing = mocker.patch.object(config_utils, "multiprocessing")
    mock_multiprocessing.cpu_count.return_value = num_cpu

    assert get_adaptive_num_workers() is None


class TestInputSizeManager:
    @pytest.mark.parametrize("base_input_size", [None, 100, [100, 200], {"train" : 100}])
    def test_init(self, base_input_size):
        mock_data_config = {"fake_data" : "fake"}
        InputSizeManager(mock_data_config, base_input_size)
