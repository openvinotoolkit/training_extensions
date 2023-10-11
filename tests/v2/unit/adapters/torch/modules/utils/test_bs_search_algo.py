from typing import Optional, Union

import pytest
from otx.v2.adapters.torch.modules.utils import BsSearchAlgo, bs_search_algo
from pytest_mock.plugin import MockerFixture


class TestBsSearchAlgo:
    @pytest.fixture(autouse=True)
    def setup_test(self, mocker: MockerFixture) -> None:
        self.mock_torch = mocker.patch.object(bs_search_algo, "torch")
        self.mock_torch.cuda.mem_get_info.return_value = (1, 10000)

    def test_init(self, mocker: MockerFixture) -> None:
        BsSearchAlgo(mocker.MagicMock(), 4, 10)

    @pytest.mark.parametrize("default_bs", [-2, 0])
    def test_init_w_wrong_default_bs(self, mocker: MockerFixture, default_bs: int) -> None:
        with pytest.raises(ValueError, match="Batch size should be bigger than 0"):
            BsSearchAlgo(mocker.MagicMock(), default_bs=default_bs, max_bs=10)

    @pytest.mark.parametrize("max_bs", [-2, 0])
    def test_init_w_wrong_max_bs(self, mocker: MockerFixture, max_bs: int) -> None:
        with pytest.raises(ValueError, match="train data set size should be bigger than 0"):
            BsSearchAlgo(mocker.MagicMock(), default_bs=4, max_bs=max_bs)

    def get_mock_train_func(self, cuda_oom_bound: int, max_runnable_bs: int) -> Union[int, float]:
        def mock_train_func(batch_size: int) -> int:
            if batch_size > cuda_oom_bound:
                mem_usage = 10000
                msg = "CUDA out of memory."
                raise RuntimeError(msg)
            if batch_size > max_runnable_bs:
                mem_usage = 8500 + 1500 * batch_size / (cuda_oom_bound - max_runnable_bs)
            else:
                mem_usage = 8500 * batch_size / max_runnable_bs

            self.mock_torch.cuda.max_memory_allocated.return_value = mem_usage
            return mem_usage

        return mock_train_func

    def test_auto_decrease_batch_size(self) -> None:
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=80)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        adapted_bs = bs_search_algo.auto_decrease_batch_size()

        assert adapted_bs == 80

    def test_find_max_usable_bs_gpu_memory_too_small(self) -> None:
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=4, max_runnable_bs=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        with pytest.raises(RuntimeError):
            bs_search_algo.auto_decrease_batch_size()

    @pytest.mark.parametrize(
        ("max_runnable_bs", "max_bs", "expected_bs"),
        [
            (100, 1000, None),
            (32, 1000, None),
            (100, 64, 64),
            (66, 1000, None),
        ],
    )
    def test_find_big_enough_batch_size(self, max_runnable_bs: int, max_bs: int, expected_bs: Optional[int]) -> None:
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=max_runnable_bs)

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, max_bs)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        if expected_bs is None:
            assert 7500 <= mock_train_func(adapted_bs) <= 8500
        else:
            assert adapted_bs == expected_bs

    def test_find_big_enough_batch_size_gpu_memory_too_small(self) -> None:
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=4, max_runnable_bs=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        with pytest.raises(RuntimeError):
            bs_search_algo.find_big_enough_batch_size()

    def test_find_big_enough_batch_size_gradient_zero(self) -> None:
        def mock_train_func(batch_size: int) -> int:
            if batch_size > 1000:
                mem_usage = 10000
                msg = "CUDA out of memory."
                raise RuntimeError(msg)
            mem_usage = 9000 if batch_size > 100 else 1000
            self.mock_torch.cuda.max_memory_allocated.return_value = mem_usage
            return mem_usage

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 1000)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        assert adapted_bs == 100

    def test_find_big_enough_batch_size_not_exceed_upper_bound(self) -> None:
        def mock_train_func(batch_size: int) -> int:
            if batch_size > 1000:
                mem_usage = 10000
                msg = "CUDA out of memory."
                raise RuntimeError(msg)
            mem_usage = 9000 if batch_size > 100 else 1000 + batch_size / 1000
            self.mock_torch.cuda.max_memory_allocated.return_value = mem_usage
            return mem_usage

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 1000)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        assert mock_train_func(adapted_bs) <= 8500

    def test_find_big_enough_batch_size_drop_last(self) -> None:
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=180)

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 200)
        adapted_bs = bs_search_algo.find_big_enough_batch_size(True)

        assert adapted_bs == 100
