from unittest.mock import MagicMock

import pytest
from otx.engine.adaptive_bs import bs_search_algo as target_file
from otx.engine.adaptive_bs.bs_search_algo import BsSearchAlgo, _get_max_memory_reserved, _get_total_memory_size


@pytest.fixture()
def mock_torch(mocker) -> MagicMock:
    return mocker.patch.object(target_file, "torch")


@pytest.fixture()
def mock_is_xpu_available(mocker) -> MagicMock:
    return mocker.patch.object(target_file, "is_xpu_available", return_value=False)


class TestBsSearchAlgo:
    @pytest.fixture(autouse=True)
    def setup_test(self, mocker, mock_torch, mock_is_xpu_available):
        self.mock_torch = mock_torch
        self.mock_torch.cuda.mem_get_info.return_value = (1, 10000)
        self.mock_mp = mocker.patch.object(target_file, "mp")

    def test_init(self, mocker):
        BsSearchAlgo(mocker.MagicMock(), 4, 10)

    @pytest.mark.parametrize("default_bs", [-2, 0])
    def test_init_w_wrong_default_bs(self, mocker, default_bs):
        with pytest.raises(ValueError, match="Batch size should be bigger than 0."):
            BsSearchAlgo(mocker.MagicMock(), default_bs=default_bs, max_bs=10)

    @pytest.mark.parametrize("max_bs", [-2, 0])
    def test_init_w_wrong_max_bs(self, mocker, max_bs):
        with pytest.raises(ValueError, match="train data set size should be bigger than 0."):
            BsSearchAlgo(mocker.MagicMock(), default_bs=4, max_bs=max_bs)

    def set_mp_process(self, train_func):
        def mock_process(target, args) -> MagicMock:
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
        def mock_train_func(batch_size) -> int:
            if batch_size > cuda_oom_bound:
                mem_usage = 10000
                msg = "CUDA out of memory."
                raise RuntimeError(msg)
            if batch_size > max_runnable_bs:
                mem_usage = 8500 + 1500 * batch_size / (cuda_oom_bound - max_runnable_bs)
            else:
                mem_usage = 8500 * batch_size / max_runnable_bs

            self.mock_torch.cuda.max_memory_reserved.return_value = mem_usage
            return mem_usage

        self.set_mp_process(mock_train_func)

        return mock_train_func

    def test_try_batch_size(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=80)
        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        batch_size = 40

        cuda_oom, max_memory_reserved = bs_search_algo._try_batch_size(batch_size)

        assert cuda_oom is False
        assert max_memory_reserved == mock_train_func(batch_size)

    def test_try_batch_size_cuda_oom(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=100, max_runnable_bs=80)
        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        batch_size = 200

        cuda_oom, _ = bs_search_algo._try_batch_size(batch_size)

        assert cuda_oom is True

    def test_auto_decrease_batch_size(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=80)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        adapted_bs = bs_search_algo.auto_decrease_batch_size()

        assert adapted_bs == 80

    def test_find_max_usable_bs_gpu_memory_too_small(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=1, max_runnable_bs=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        with pytest.raises(RuntimeError):
            bs_search_algo.auto_decrease_batch_size()

    def test_auto_decrease_batch_size_bs2_not_oom_but_most_mem(self):
        """Batch size 2 doesn't make oom but use most of memory."""
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=2, max_runnable_bs=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        assert bs_search_algo.auto_decrease_batch_size() == 2

    @pytest.mark.parametrize(
        ("max_runnable_bs", "max_bs", "expected_bs"),
        [
            (100, 1000, None),
            (32, 1000, None),
            (100, 64, 64),
            (66, 1000, None),
        ],
    )
    def test_find_big_enough_batch_size(self, max_runnable_bs, max_bs, expected_bs):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=max_runnable_bs)

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, max_bs)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        if expected_bs is None:
            assert 7500 <= mock_train_func(adapted_bs) <= 8500
        else:
            assert adapted_bs == expected_bs

    def test_find_big_enough_batch_size_gpu_memory_too_small(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=1, max_runnable_bs=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        with pytest.raises(RuntimeError):
            bs_search_algo.find_big_enough_batch_size()

    def test_find_big_enough_batch_size_bs2_not_oom_but_most_mem(self):
        """Batch size 2 doesn't make oom but use most of memory."""
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=2, max_runnable_bs=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, 2, 1000)
        assert bs_search_algo.find_big_enough_batch_size() == 2

    def test_find_big_enough_batch_size_gradient_zero(self):
        def mock_train_func(batch_size) -> int:
            if batch_size > 1000:
                mem_usage = 10000
                msg = "CUDA out of memory."
                raise RuntimeError(msg)
            mem_usage = 9000 if batch_size > 100 else 1000
            self.mock_torch.cuda.max_memory_reserved.return_value = mem_usage
            return mem_usage

        self.set_mp_process(mock_train_func)

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 1000)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        assert adapted_bs == 100

    def test_find_big_enough_batch_size_not_exceed_upper_bound(self):
        def mock_train_func(batch_size) -> int:
            if batch_size > 1000:
                mem_usage = 10000
                msg = "CUDA out of memory."
                raise RuntimeError(msg)
            mem_usage = 9000 if batch_size > 100 else 1000 + batch_size / 1000
            self.mock_torch.cuda.max_memory_reserved.return_value = mem_usage
            return mem_usage

        self.set_mp_process(mock_train_func)

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 1000)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        assert mock_train_func(adapted_bs) <= 8500

    def test_find_big_enough_batch_size_drop_last(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=180)

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 200)
        adapted_bs = bs_search_algo.find_big_enough_batch_size(True)

        assert adapted_bs == 100


def test_get_max_memory_reserved(mock_torch, mock_is_xpu_available):
    _get_max_memory_reserved()
    mock_torch.cuda.max_memory_reserved.assert_called_once()


def test_get_max_xpu_memory_reserved(mock_torch, mock_is_xpu_available):
    mock_is_xpu_available.return_value = True
    _get_max_memory_reserved()
    mock_torch.xpu.max_memory_reserved.assert_called_once()


def test_get_total_memory_size(mock_torch, mock_is_xpu_available):
    total_mem = 100
    mock_torch.cuda.mem_get_info.return_value = (1, total_mem)
    assert _get_total_memory_size() == total_mem


def test_get_total_xpu_memory_size(mock_torch, mock_is_xpu_available):
    mock_is_xpu_available.return_value = True
    total_mem = 100
    mock_torch.xpu.get_device_properties.return_value.total_memory = total_mem
    assert _get_total_memory_size() == total_mem
