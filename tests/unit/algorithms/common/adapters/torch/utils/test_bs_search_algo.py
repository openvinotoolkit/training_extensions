from typing import Optional, List

import pytest
import torch

from tests.test_suite.e2e_test_system import e2e_pytest_unit
from otx.algorithms.common.adapters.torch.utils import BsSearchAlgo
from otx.algorithms.common.adapters.torch.utils import bs_search_algo


@e2e_pytest_unit
class TestBsSearchAlgo:
    @pytest.fixture(autouse=True)
    def setup_test(self, mocker):
        self.mock_torch = mocker.patch.object(bs_search_algo, "torch")
        self.mock_torch.cuda.mem_get_info.return_value = (1, 10000)
        self.mock_dist = mocker.patch.object(bs_search_algo, "dist")
        self.mock_dist.is_initialized.return_value = False

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

        return mock_train_func

    def test_try_batch_size(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=80)
        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        batch_size = 40

        cuda_oom, max_memory_reserved = bs_search_algo._try_batch_size(batch_size)

        assert cuda_oom is False
        assert max_memory_reserved == mock_train_func(batch_size)
        self.mock_torch.cuda.reset_max_memory_cached.assert_called()
        self.mock_torch.cuda.empty_cache.assert_called()

    def test_try_batch_size_cuda_oom(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=100, max_runnable_bs=80)
        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        batch_size = 200

        cuda_oom, _ = bs_search_algo._try_batch_size(batch_size)

        assert cuda_oom is True
        self.mock_torch.cuda.reset_max_memory_cached.assert_called()
        self.mock_torch.cuda.empty_cache.assert_called()

    def _prepare_dist_test(self, broadcast_val: torch.Tensor, gather_val: Optional[List[torch.Tensor]] = None):
        self.mock_dist.is_initialized.return_value = True

        # mocking torch.distributed.broadcast
        def mock_broadcast(tensor: torch.Tensor, src: int):
            tensor.copy_(broadcast_val)

        self.mock_dist.broadcast.side_effect = mock_broadcast

        # mocking torch.distributed.gather if gather_val is given
        def mock_gather(tensor: torch.Tensor, gather_list: Optional[List[torch.Tensor]] = None, dst: int = 0):
            for i in range(len(gather_list)):
                gather_list[i].copy_(gather_val[i])

        if gather_val is not None:
            self.mock_dist.gather.side_effect = mock_gather

        # revert some of torch function
        def mock_tensor_cuda(self, *args, **kwargs):
            return self

        torch.Tensor.cuda = mock_tensor_cuda
        self.mock_torch.tensor = torch.tensor
        self.mock_torch.int64 = torch.int64
        self.mock_torch.max = torch.max
        self.mock_torch.any = torch.any
        self.mock_torch.stack = torch.stack
        self.mock_torch.empty = torch.empty

    def test_try_batch_size_distributed_not_rank_0(self):
        self.mock_dist.get_rank.return_value = 1
        broadcasted_cuda_oom = False
        broadcasted_max_memory_reserved = 4000
        self._prepare_dist_test(
            broadcast_val=torch.tensor([broadcasted_cuda_oom, broadcasted_max_memory_reserved], dtype=torch.int64)
        )
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=80)
        batch_size = 40
        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        w1_max_memory_reserved = mock_train_func(batch_size)

        cuda_oom, max_memory_reserved = bs_search_algo._try_batch_size(batch_size)

        # check dist.gather is called and get [cuda_oom, maxmemory_reserved] as arguments.
        self.mock_dist.gather.assert_called_once()
        assert self.mock_dist.gather.call_args.args[0][0].item() == False
        assert self.mock_dist.gather.call_args.args[0][1].item() == w1_max_memory_reserved
        assert self.mock_dist.gather.call_args.kwargs["dst"] == 0
        # check dist.broadcast is called
        self.mock_dist.broadcast.assert_called_once()
        assert self.mock_dist.broadcast.call_args.kwargs["src"] == 0
        # check broadcased values are returned
        assert cuda_oom is broadcasted_cuda_oom
        assert max_memory_reserved == broadcasted_max_memory_reserved

    def test_try_batch_size_distributed_rank_0(self):
        self.mock_dist.get_rank.return_value = 0
        self.mock_dist.get_world_size.return_value = 2
        self._prepare_dist_test(
            broadcast_val=torch.tensor([True, 4000], dtype=torch.int64),
            gather_val=[
                torch.tensor([False, 3000], dtype=torch.int64),
                torch.tensor([True, 4000], dtype=torch.int64),
            ],
        )
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=80)
        batch_size = 40
        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        w0_max_memory_reserved = mock_train_func(batch_size)

        cuda_oom, max_memory_reserved = bs_search_algo._try_batch_size(batch_size)

        # check dist.gather is called and get [cuda_oom, max_memory_reserved] as arguments.
        self.mock_dist.gather.assert_called_once()
        assert self.mock_dist.gather.call_args.args[0][0].item() == False
        assert self.mock_dist.gather.call_args.args[0][1].item() == w0_max_memory_reserved
        assert self.mock_dist.gather.call_args.kwargs["dst"] == 0
        # check if any process get cuda oom then set cuda_oom to True and
        # set max_memory_reserved to maximum value of processes'
        self.mock_dist.broadcast.assert_called_once()
        self.mock_dist.broadcast.assert_called_once()
        assert self.mock_dist.broadcast.call_args.kwargs["src"] == 0
        assert self.mock_dist.broadcast.call_args.args[0][0].item() == True
        assert self.mock_dist.broadcast.call_args.args[0][1].item() == 4000
        # check proper values are returned
        assert cuda_oom is True
        assert max_memory_reserved == 4000

    def test_auto_decrease_batch_size(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=80)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        adapted_bs = bs_search_algo.auto_decrease_batch_size()

        assert adapted_bs == 80

    def test_find_max_usable_bs_gpu_memory_too_small(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=4, max_runnable_bs=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
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
    def test_find_big_enough_batch_size(self, max_runnable_bs, max_bs, expected_bs):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=max_runnable_bs)

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, max_bs)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        if expected_bs is None:
            assert 7500 <= mock_train_func(adapted_bs) <= 8500
        else:
            assert adapted_bs == expected_bs

    def test_find_big_enough_batch_size_gpu_memory_too_small(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=4, max_runnable_bs=1)

        bs_search_algo = BsSearchAlgo(mock_train_func, 128, 1000)
        with pytest.raises(RuntimeError):
            bs_search_algo.find_big_enough_batch_size()

    def test_find_big_enough_batch_size_gradient_zero(self):
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

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 1000)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        assert adapted_bs == 100

    def test_find_big_enough_batch_size_not_exceed_upper_bound(self):
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

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 1000)
        adapted_bs = bs_search_algo.find_big_enough_batch_size()

        assert mock_train_func(adapted_bs) <= 8500

    def test_find_big_enough_batch_size_drop_last(self):
        mock_train_func = self.get_mock_train_func(cuda_oom_bound=10000, max_runnable_bs=180)

        bs_search_algo = BsSearchAlgo(mock_train_func, 64, 200)
        adapted_bs = bs_search_algo.find_big_enough_batch_size(True)

        assert adapted_bs == 100
