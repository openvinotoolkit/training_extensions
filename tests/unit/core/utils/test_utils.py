import pytest
from otx.core.utils import utils as target_file
from otx.core.utils.utils import get_adaptive_num_workers


@pytest.mark.parametrize("num_dataloader", [1, 2, 4])
def test_get_adaptive_num_workers(mocker, num_dataloader):
    num_gpu = 5
    mock_torch = mocker.patch.object(target_file, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mocker.patch.object(target_file, "cpu_count", return_value=num_cpu)

    assert get_adaptive_num_workers(num_dataloader) == num_cpu // (num_gpu * num_dataloader)


def test_get_adaptive_num_workers_no_gpu(mocker):
    num_gpu = 0
    mock_torch = mocker.patch.object(target_file, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mocker.patch.object(target_file, "cpu_count", return_value=num_cpu)

    assert get_adaptive_num_workers() is None
