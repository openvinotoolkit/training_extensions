import pytest

from otx.algorithms.common.adapters.torch.utils import BsSearchAlgo
from otx.algorithms.common.adapters.torch.utils import bs_search_algo


class TestBsSearchAlgo:
    @pytest.fixture(autouse=True)
    def setup_test(self, mocker):
        self.mock_torch = mocker.patch.object(bs_search_algo, "torch")
        self.mock_torch.cuda.mem_get_info.return_value = (1, 10000)

    def test_init(self, mocker):
        BsSearchAlgo(mocker.MagicMock(), 4, 10)

    @pytest.mark.parametrize("default_bs", [-2, 0])
    def test_init_w_wrong_default_bs(self, mocker, default_bs):
        with pytest.raises(ValueError):
            BsSearchAlgo(mocker.MagicMock(), default_bs=default_bs, max_bs=10)

    @pytest.mark.parametrize("max_bs", [-2, 0])
    def test_init_w_wrong_default_bs(self, mocker, max_bs):
        with pytest.raises(ValueError):
            BsSearchAlgo(mocker.MagicMock(), default_bs=4, max_bs=max_bs)

    def get_mock_train_func(self, cuda_oom_bound: int, use_much_mom_bound: int):
        def mock_train_func(batch_size):
            if batch_size > cuda_oom_bound:
                self.mock_torch.cuda.max_memory_allocated.return_value = 10000
                raise RuntimeError("CUDA out of memory.")
            elif batch_size > use_much_mom_bound:
                self.mock_torch.cuda.max_memory_allocated.return_value = 9000
            else:
                self.mock_torch.cuda.max_memory_allocated.return_value = 1000

        return mock_train_func

    def test_auto_decrease_batch_size(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, use_much_mom_bound=80)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        adapted_bs = bs_search_algo.auto_decrease_batch_size()

        assert adapted_bs == 80

    def test_find_max_usable_bs_gpu_memory_too_small(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=4, use_much_mom_bound=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        with pytest.raises(RuntimeError):
            bs_search_algo.auto_decrease_batch_size()

    def test_find_big_enough_batch_size(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, use_much_mom_bound=100)

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 1000)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        assert adapted_bs == 100

    def test_find_big_enough_batch_size_default_unrunnable(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, use_much_mom_bound=32)

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 1000)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        assert adapted_bs == 32

    def test_find_big_enough_batch_size_gpu_memory_too_small(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=4, use_much_mom_bound=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        with pytest.raises(RuntimeError):
            bs_search_algo.find_big_enough_batch_size()

    def test_find_big_enough_batch_size_default_bs_is_max_bs(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, use_much_mom_bound=100)

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 64)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        assert adapted_bs == 64
