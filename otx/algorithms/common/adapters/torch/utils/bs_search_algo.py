"""Algorithm to find a proper batch size which is fit to current GPU device."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Tuple

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
        self._mem_upper_bound = 0.85 * self._total_mem

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
            if cuda_oom or max_memory_allocated > self._mem_upper_bound:
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

    def find_big_enough_batch_size(self, drop_last: bool = False) -> int:
        """Find a big enough batch size.

        This function finds a big enough batch size by training with various batch sizes.
        It estimate a batch size using equation is estimated using training history.
        The reason why using the word "big enough" is that it tries to find not maxmium but big enough value which uses
        GPU memory between lower and upper bound.

        Args:
            drop_last (bool): Whether to drop the last incomplete batch.

        Raises:
            RuntimeError: If training with batch size 2 can't be run, raise an error.

        Returns:
            int: Big enough batch size.
        """
        estimated_bs = self._default_bs

        # try default batch size
        cuda_oom, bs_mem_usage = self._try_batch_size(estimated_bs)
        if cuda_oom or bs_mem_usage > self._mem_upper_bound:
            self._default_bs -= 2
            if self._default_bs <= 0:
                raise RuntimeError("Current device can't train model even with 2.")

            return self.auto_decrease_batch_size()

        # try default batch size + 2
        estimated_bs += 2
        if estimated_bs > self._max_bs:
            return self._default_bs
        cuda_oom, bs_mem_usage = self._try_batch_size(estimated_bs)
        if cuda_oom or bs_mem_usage > self._mem_upper_bound:
            return self._default_bs

        # estimate batch size using equation
        estimation_pct = 0.82
        while True:
            estimated_bs = self._estimate_batch_size(estimation_pct)
            if estimated_bs in self._bs_try_history:
                break
            cuda_oom, mem_usage = self._try_batch_size(estimated_bs)

            if cuda_oom:
                estimation_pct -= 0.1
                if estimation_pct <= 0:
                    estimated_bs = self._default_bs + 2
                    break
            elif self._mem_lower_bound <= mem_usage <= self._mem_upper_bound:
                break
            else:
                estimation_pct = 0.82

        if drop_last and (self._max_bs // 2 < estimated_bs < self._max_bs):
            estimated_bs = self._max_bs // 2

        return estimated_bs

    def _estimate_batch_size(self, estimation_pct: float) -> int:
        if len(self._bs_try_history) < 2:
            raise RuntimeError("At least two trials should be done without CUDA OOM to estimate batch size.")

        def distance_from_bound(val):
            if val[1] < self._mem_lower_bound:
                # if memory usage is same, then higher batch size is preferred
                return self._mem_lower_bound - val[1] - val[0] / 10000
            elif self._mem_upper_bound < val[1]:
                # if memory usage is same, then lower batch size is preferred
                return val[1] - self._mem_upper_bound + val[0] / 10000
            else:
                return 0

        bs_arr = sorted([(bs, mem_usage) for bs, mem_usage in self._bs_try_history.items()], key=distance_from_bound)
        bs1 = bs_arr[0][0]
        bs1_mem_usage = bs_arr[0][1]

        for i in range(1, len(bs_arr)):
            graident = (bs_arr[i][1] - bs1_mem_usage) / (bs_arr[i][0] - bs1)
            b = bs1_mem_usage - graident * bs1
            if graident != 0:
                break

        if graident == 0:  # all batch size history used same GPU memory
            if bs1_mem_usage < self._mem_lower_bound:
                return bs1 + 2
            elif bs1_mem_usage > self._mem_upper_bound:
                if bs1 <= 2:
                    return 2
                return bs1 - 2
            else:
                return bs1

        estimated_bs = round(((self._total_mem * estimation_pct) - b) / (graident * 2)) * 2

        # If estimated_bs is already tried and it used GPU memory more than upper bound,
        # set estimated_bs as lowest value of batch sizes using GPU memory more than uppoer bound - 2
        if estimated_bs in self._bs_try_history and self._bs_try_history[estimated_bs] > self._mem_upper_bound:
            for bs, mem_usage in bs_arr:
                if mem_usage > self._mem_upper_bound:
                    estimated_bs = bs - 2
                    break

        if estimated_bs > self._max_bs:
            estimated_bs = self._max_bs

        return estimated_bs
