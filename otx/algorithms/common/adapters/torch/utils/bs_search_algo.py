"""Algorithm to find a proper batch size which is fit to current GPU device."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Tuple, List, Dict

import torch

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


class BsSearchAlgo:
    """Algorithm class to find optimal batch size.

    Args:
        train_func (Callable[[int], None]): Training function with single arugment to set batch size.
        default_bs (int): Default batch size. It should be bigger than 0.
        max_bs (int): Maximum batch size. It should be bigger than 0.
    """

    def __init__(self, train_func: Callable[[int], None], default_bs: int, max_bs: int):
        if default_bs <= 0:
            raise ValueError("Batch size should be bigger than 0.")
        if max_bs <= 0:
            raise ValueError("train data set size should be bigger than 0.")

        if max_bs < default_bs:
            default_bs = max_bs

        self._train_func = train_func
        self._default_bs = default_bs
        self._max_bs = max_bs
        self._bs_try_history: Dict[int, int] = {}
        _, self._total_mem = torch.cuda.mem_get_info()
        self._mem_lower_bound = 0.8 * self._total_mem
        self._mem_higher_bound = 0.85 * self._total_mem

    def _try_batch_size(self, bs: int) -> Tuple[bool, int]:
        cuda_oom = False
        torch.cuda.reset_max_memory_allocated(device=None)
        torch.cuda.empty_cache()

        try:
            self._train_func(bs)
        except RuntimeError as e:
            if str(e).startswith("CUDA out of memory."):
                cuda_oom = True
            else:
                raise e

        max_memory_allocated = torch.cuda.max_memory_allocated(device=None)
        if not cuda_oom:
            # Because heapq only supports min heap, use negatized batch size
            self._bs_try_history[bs] = max_memory_allocated

        logger.debug(
            f"Adapting Batch size => bs : {bs}, CUDA_OOM : {cuda_oom}, "
            f"GPU memory usage : {max_memory_allocated / self._total_mem}%"
        )
        torch.cuda.empty_cache()

        return cuda_oom, max_memory_allocated

    @staticmethod
    def _get_even_center_val(val1: int, val2: int) -> int:
        ret = (val1 + val2) // 2
        if ret % 2 == 1:
            ret += 1
        return ret

    def auto_decrease_batch_size(self) -> int:
        """Decrease batch size if default batch size isn't fit to current GPU device.

        Returns:
            int: Proper batch size possibly decreased as default value isn't fit
        """
        available_bs = 0
        current_bs = self._default_bs
        lowest_unavailable_bs = self._default_bs + 2

        while True:
            cuda_oom, max_memory_allocated = self._try_batch_size(current_bs)

            # If GPU memory usage is too close to limit, CUDA OOM can be raised during training
            if cuda_oom or self._mem_higher_bound < max_memory_allocated:
                if current_bs < lowest_unavailable_bs:
                    lowest_unavailable_bs = current_bs
                current_bs = self._get_even_center_val(current_bs, available_bs)
            else:
                available_bs = current_bs
                current_bs = self._get_even_center_val(current_bs, lowest_unavailable_bs)

            if lowest_unavailable_bs - available_bs <= 2:
                break

        if available_bs == 0:
            raise RuntimeError("Current device can't train model even with 2.")

        return available_bs

    def find_big_enough_batch_size(self) -> int:
        current_bs = self._default_bs

        # try default batch size
        cuda_oom, bs_mem_usage = self._try_batch_size(current_bs)
        if cuda_oom or self._mem_higher_bound < bs_mem_usage:
            self._default_bs -= 2
            if self._default_bs <= 0:
                raise RuntimeError("Current device can't train model even with 2.")

            return self.auto_decrease_batch_size()

        # try default batch size + 2
        current_bs += 2
        cuda_oom, bs_mem_usage = self._try_batch_size(current_bs)
        if cuda_oom or self._mem_higher_bound < bs_mem_usage:
            return  current_bs

        estimation_pct = 0.8
        while True:
            current_bs = self._estimate_batch_size(estimation_pct)
            if current_bs > self._max_bs:
                current_bs = self._max_bs
            if current_bs in self._bs_try_history:
                return current_bs
            cuda_oom, mem_usage = self._try_batch_size(current_bs)

            if cuda_oom:
                estimation_pct -= 0.1
                if estimation_pct <= 0:
                    return self._default_bs + 2
            elif self._mem_lower_bound <= mem_usage <= self._mem_higher_bound:
                return current_bs

    def _estimate_batch_size(self, estimation_pct: float) -> int:
        if len(self._bs_try_history) < 2:
            raise RuntimeError("At least two trials should be done without CUDA OOM to estimate batch size.")

        def distance_from_bound(val):
            if val[1] < self._mem_lower_bound:
                return self._mem_lower_bound - val[1]
            elif self._mem_higher_bound < val[1]:
                return val[1] - self._mem_higher_bound
            else:
                return 0

        bs_arr = sorted([(bs, mem_usage) for bs, mem_usage in self._bs_try_history.items()], key=distance_from_bound)
        bs1 = bs_arr[0][0]
        bs1_mem_usage = bs_arr[0][1]

        graident = (bs_arr[1][1] - bs_arr[0][1]) / (bs_arr[1][0] - bs_arr[0][0])
        b = bs1_mem_usage - graident * bs1

        return round(((self._total_mem * estimation_pct) - b) / (graident * 2)) * 2
