"""Algorithm to find a proper batch size which is fit to current GPU device."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Tuple

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
        _, self._total_mem = torch.cuda.mem_get_info()

    def _try_batch_size(self, bs: int) -> Tuple[bool, int]:
        cuda_oom = False
        torch.cuda.reset_max_memory_allocated(device=None)

        try:
            self._train_func(bs)
        except RuntimeError as e:
            if str(e).startswith("CUDA out of memory."):
                cuda_oom = True
            else:
                raise e

        max_memory_allocated = torch.cuda.max_memory_allocated(device=None)

        logger.debug(
            f"Adapting Batch size => bs : {bs}, CUDA_OOM : {cuda_oom}, "
            f"GPU memory usage : {max_memory_allocated / self._total_mem}%"
        )

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
            if cuda_oom or max_memory_allocated >= self._total_mem * 0.85:
                if current_bs < lowest_unavailable_bs:
                    lowest_unavailable_bs = current_bs
                current_bs = self._get_even_center_val(current_bs, available_bs)
            else:
                available_bs = current_bs
                current_bs = self._get_even_center_val(current_bs, lowest_unavailable_bs)

            torch.cuda.empty_cache()
            if lowest_unavailable_bs - available_bs <= 2:
                break

        if available_bs == 0:
            raise RuntimeError("Current device can't train model even with 2.")

        return available_bs

    def find_big_enough_batch_size(self) -> int:
        current_bs = self._default_bs

        # try default batch size
        cuda_oom, max_memory_allocated = self._try_batch_size(current_bs)
        if cuda_oom:
            self._default_bs -= 2
            if self._default_bs <= 0:
                raise RuntimeError("Current device can't train model even with 2.")

            return self.auto_decrease_batch_size()

        # try default batch size + 2
        current_bs += 2
        cuda_oom, max_memory_allocated = self._try_batch_size(current_bs)
