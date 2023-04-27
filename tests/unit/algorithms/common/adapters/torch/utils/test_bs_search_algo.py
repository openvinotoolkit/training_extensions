import pytest

from otx.algorithms.common.adapters.torch.utils import find_max_usable_bs
from otx.algorithms.common.adapters.torch.utils import bs_search_algo


def test_find_max_usable_bs(mocker):
    mocker_torch = mocker.patch.object(bs_search_algo, "torch")
    mocker_torch.cuda.mem_get_info.return_value = (1, 10000)

    def mock_train_func(batch_size):
        if batch_size > 100:
            mocker_torch.cuda.max_memory_allocated.return_value = 10000
            raise RuntimeError("CUDA out of memory.")
        elif batch_size > 80:
            mocker_torch.cuda.max_memory_allocated.return_value = 9000
        else:
            mocker_torch.cuda.max_memory_allocated.return_value = 1000

    adapted_bs = find_max_usable_bs(mock_train_func, 128, 1000)

    assert adapted_bs == 80


def test_find_max_usable_bs_gpu_memory_too_small(mocker):
    mocker_torch = mocker.patch.object(bs_search_algo, "torch")
    mocker_torch.cuda.mem_get_info.return_value = (1, 10000)

    def mock_train_func(batch_size):
        if batch_size > 4:
            mocker_torch.cuda.max_memory_allocated.return_value = 10000
            raise RuntimeError("CUDA out of memory.")
        elif batch_size >= 2:
            mocker_torch.cuda.max_memory_allocated.return_value = 9000
        else:
            mocker_torch.cuda.max_memory_allocated.return_value = 1000

    with pytest.raises(RuntimeError):
        find_max_usable_bs(mock_train_func, 128, 1000)


@pytest.mark.parametrize("default_bs", [-1, 0])
def test_find_max_usable_bs_wrong_default_bs(mocker, default_bs):
    with pytest.raises(ValueError):
        find_max_usable_bs(mocker.MagicMock(), default_bs, 1000)


@pytest.mark.parametrize("trainset_size", [-1, 0])
def test_find_max_usable_bs_wrong_trainset_size(mocker, trainset_size):
    with pytest.raises(ValueError):
        find_max_usable_bs(mocker.MagicMock(), 8, trainset_size)
