from unittest.mock import MagicMock
from typing import Optional, Callable

import pytest

from tests.test_suite.e2e_test_system import e2e_pytest_unit
from otx.algorithms.common.adapters.torch.utils import BsSearchAlgo
from otx.algorithms.common.adapters.torch.utils import bs_search_algo


@pytest.fixture
def train_func_kwargs():
    return MagicMock()


@e2e_pytest_unit
class TestBsSearchAlgo:
    @pytest.fixture(autouse=True)
    def setup_test(self, mocker):
        self.mock_torch = mocker.patch.object(bs_search_algo, "torch")
        self.mock_torch.cuda.mem_get_info.return_value = (1, 10000)
        self.mock_mp = mocker.patch.object(bs_search_algo, "mp")
        mocker.patch.object(bs_search_algo, "is_xpu_available", return_value=False)

    def test_init(self, mocker, train_func_kwargs):
        BsSearchAlgo(mocker.MagicMock(), train_func_kwargs, 4, 10)

    @pytest.mark.parametrize("default_bs", [-2, 0])
    def test_init_w_wrong_default_bs(self, mocker, default_bs, train_func_kwargs):
        with pytest.raises(ValueError):
            BsSearchAlgo(mocker.MagicMock(), train_func_kwargs, default_bs=default_bs, max_bs=10)

    @pytest.mark.parametrize("max_bs", [-2, 0])
    def test_init_w_wrong_default_bs(self, mocker, max_bs, train_func_kwargs):
        with pytest.raises(ValueError):
            BsSearchAlgo(mocker.MagicMock(), train_func_kwargs, default_bs=4, max_bs=max_bs)

    def set_mp_process(self, train_func):
        def mock_process(target, args):
            batch_size = args[-2]
            oom = False
            mem_usage = 0

            try:
                mem_usage = train_func(batch_size)
            except RuntimeError:
                oom = True

            trial_queue = args[-1]
            trial_queue.get.return_value = {"oom": oom, "max_memory_reserved": mem_usage}

            return MagicMock()

        self.mock_mp.get_context.return_value.Process.side_effect = mock_process

    def get_mock_train_func(self, cuda_oom_bound: int, max_runnable_bs: int):
        def mock_train_func(batch_size):
            if batch_size > cuda_oom_bound:
                mem_usage = 10000
                raise RuntimeError("CUDA out of memory.")
            elif batch_size > max_runnable_bs:
                mem_usage = 8500 + 1500 * batch_size / (cuda_oom_bound - max_runnable_bs)
            else:
                mem_usage = 8500 * batch_size / max_runnable_bs

            self.mock_torch.cuda.max_memory_reserved.return_value = mem_usage
            return mem_usage

        self.set_mp_process(mock_train_func)

        return mock_train_func

    def test_try_batch_size(self, train_func_kwargs):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=80)
        bs_search_algo = BsSearchAlgo(mock_train_func, train_func_kwargs, 128, 1000)
        batch_size = 40

        cuda_oom, max_memory_reserved = bs_search_algo._try_batch_size(batch_size)

        assert cuda_oom is False
        assert max_memory_reserved == mock_train_func(batch_size)

    def test_try_batch_size_cuda_oom(self, train_func_kwargs):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=100, max_runnable_bs=80)
        bs_search_algo = BsSearchAlgo(mock_train_func, train_func_kwargs, 128, 1000)
        batch_size = 200

        cuda_oom, _ = bs_search_algo._try_batch_size(batch_size)

        assert cuda_oom is True

    def test_auto_decrease_batch_size(self, train_func_kwargs):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=80)

        bs_search_algo = BsSearchAlgo(mock_train_func, train_func_kwargs, 128, 1000)
        adapted_bs = bs_search_algo.auto_decrease_batch_size()

        assert adapted_bs == 80

    def test_find_max_usable_bs_gpu_memory_too_small(self, train_func_kwargs):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=4, max_runnable_bs=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, train_func_kwargs, 128, 1000)
        with pytest.raises(RuntimeError):
            bs_search_algo.auto_decrease_batch_size()

    @pytest.mark.parametrize(
        "max_runnable_bs,max_bs,expected_bs",
        [
            (100, 1000, None),
            (32, 1000, None),
            (100, 64, 64),
            (66, 1000, None),
        ],
    )
    def test_find_big_enough_batch_size(self, max_runnable_bs, max_bs, expected_bs, train_func_kwargs):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=max_runnable_bs)

        bs_search_algo = BsSearchAlgo(mock_train_func, train_func_kwargs, 64, max_bs)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        if expected_bs is None:
            assert 7500 <= mock_train_func(adapted_bs) <= 8500
        else:
            assert adapted_bs == expected_bs

    def test_find_big_enough_batch_size_gpu_memory_too_small(self, train_func_kwargs):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=4, max_runnable_bs=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, train_func_kwargs, 128, 1000)
        with pytest.raises(RuntimeError):
            bs_search_algo.find_big_enough_batch_size()

    def test_find_big_enough_batch_size_gradient_zero(self, train_func_kwargs):
        def mock_train_func(batch_size):
            if batch_size > 1000:
                mem_usage = 10000
                raise RuntimeError("CUDA out of memory.")
            elif batch_size > 100:
                mem_usage = 9000
            else:
                mem_usage = 1000
            self.mock_torch.cuda.max_memory_reserved.return_value = mem_usage
            return mem_usage

        self.set_mp_process(mock_train_func)

        bs_search_algo = BsSearchAlgo(mock_train_func, train_func_kwargs, 64, 1000)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        assert adapted_bs == 100

    def test_find_big_enough_batch_size_not_exceed_upper_bound(self, train_func_kwargs):
        def mock_train_func(batch_size):
            if batch_size > 1000:
                mem_usage = 10000
                raise RuntimeError("CUDA out of memory.")
            elif batch_size > 100:
                mem_usage = 9000
            else:
                mem_usage = 1000 + batch_size / 1000
            self.mock_torch.cuda.max_memory_reserved.return_value = mem_usage
            return mem_usage

        self.set_mp_process(mock_train_func)

        bs_search_algo = BsSearchAlgo(mock_train_func, train_func_kwargs, 64, 1000)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        assert mock_train_func(adapted_bs) <= 8500

    def test_find_big_enough_batch_size_drop_last(self, train_func_kwargs):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=180)

        bs_search_algo = BsSearchAlgo(mock_train_func, train_func_kwargs, 64, 200)
        adapted_bs = bs_search_algo.find_big_enough_batch_size(True)

        assert adapted_bs == 100
