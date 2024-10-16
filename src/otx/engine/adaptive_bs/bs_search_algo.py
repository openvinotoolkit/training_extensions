# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Algorithm to find a proper batch size which is fit to current device."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from typing import Any, Callable

import torch

from otx.utils.utils import is_xpu_available

logger = logging.getLogger(__name__)


class BsSearchAlgo:
    """Algorithm class to find optimal batch size.

    Args:
        train_func (Callable[[int], Any]): Training function with single arugment to set batch size.
        default_bs (int): Default batch size. It should be bigger than 0.
        max_bs (int): Maximum batch size. It should be bigger than 0.
    """

    def __init__(
        self,
        train_func: Callable[[int], Any],
        default_bs: int,
        max_bs: int,
    ):
        if default_bs <= 0:
            msg = "Batch size should be bigger than 0."
            raise ValueError(msg)
        if max_bs <= 0:
            msg = "train data set size should be bigger than 0."
            raise ValueError(msg)

        if max_bs < default_bs:
            default_bs = max_bs

        self._train_func = train_func
        self._default_bs = default_bs
        self._max_bs = max_bs
        self._bs_try_history: dict[int, int] = {}
        self._total_mem = _get_total_memory_size()
        self._mem_lower_bound = 0.8 * self._total_mem
        self._mem_upper_bound = 0.85 * self._total_mem
        self._mp_ctx = mp.get_context("spawn")

    def _try_batch_size(self, bs: int) -> tuple[bool, int]:
        trial_queue = self._mp_ctx.Queue()
        proc = self._mp_ctx.Process(target=_run_trial, args=(self._train_func, bs, trial_queue))
        proc.start()
        output = None
        while proc.is_alive():
            try:
                output = trial_queue.get(timeout=1)
                break
            except queue.Empty:
                pass
        proc.join()
        if output is None:
            msg = "There is no output from the trial for adaptive batch size."
            raise RuntimeError(msg)

        oom = output["oom"]
        max_memory_reserved = output["max_memory_reserved"]

        if not oom:
            self._bs_try_history[bs] = max_memory_reserved

        logger.debug(
            f"Adapting Batch size => bs : {bs}, OOM : {oom}, "
            f"memory usage : {max_memory_reserved / self._total_mem}%",
        )

        return oom, max_memory_reserved

    @staticmethod
    def _get_even_center_val(val1: int, val2: int) -> int:
        ret = (val1 + val2) // 2
        if ret % 2 == 1:
            ret += 1
        return ret

    def auto_decrease_batch_size(self) -> int:
        """Decrease batch size if default batch size isn't fit to current device.

        Returns:
            int: Proper batch size possibly decreased as default value isn't fit
        """
        available_bs = 0
        current_bs = self._default_bs
        lowest_unavailable_bs = self._default_bs + 2

        while True:
            oom, max_memory_reserved = self._try_batch_size(current_bs)

            # If memory usage is too close to limit, OOM can be raised during training
            if oom or max_memory_reserved > self._mem_upper_bound:
                if current_bs < lowest_unavailable_bs:
                    lowest_unavailable_bs = current_bs
                current_bs = self._get_even_center_val(current_bs, available_bs)
            else:
                available_bs = current_bs
                current_bs = self._get_even_center_val(current_bs, lowest_unavailable_bs)

            if lowest_unavailable_bs - available_bs <= 2:
                break

        if available_bs == 0:
            if oom:
                msg = "Current device can't train model even with 2."
                raise RuntimeError(msg)
            logger.warning(
                "Even with a batch size of 2, most of the memory is used, "
                "which could cause the training to fail midway.",
            )
            available_bs = 2

        return available_bs

    def find_big_enough_batch_size(self, drop_last: bool = False) -> int:
        """Find a big enough batch size.

        This function finds a big enough batch size by training with various batch sizes.
        It estimate a batch size using equation is estimated using training history.
        The reason why using the word "big enough" is that it tries to find not maxmium but big enough value which uses
        memory between lower and upper bound.

        Args:
            drop_last (bool): Whether to drop the last incomplete batch.

        Raises:
            RuntimeError: If training with batch size 2 can't be run, raise an error.

        Returns:
            int: Big enough batch size.
        """
        estimated_bs = self._default_bs

        # try default batch size
        oom, bs_mem_usage = self._try_batch_size(estimated_bs)
        if oom or bs_mem_usage > self._mem_upper_bound:
            self._default_bs -= 2
            if self._default_bs <= 0:
                if oom:
                    msg = "Current device can't train model even with 2."
                    raise RuntimeError(msg)
                logger.warning(
                    "Even with a batch size of 2, most of the memory is used, "
                    "which could cause the training to fail midway.",
                )
                return 2

            return self.auto_decrease_batch_size()

        # try default batch size + 2
        estimated_bs += 2
        if estimated_bs > self._max_bs:
            return self._default_bs
        oom, bs_mem_usage = self._try_batch_size(estimated_bs)
        if oom or bs_mem_usage > self._mem_upper_bound:
            return self._default_bs

        # estimate batch size using equation
        estimation_pct = 0.82
        while True:
            estimated_bs = self._estimate_batch_size(estimation_pct)
            if estimated_bs in self._bs_try_history:
                break
            oom, mem_usage = self._try_batch_size(estimated_bs)

            if oom:
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
            msg = "At least two trials should be done without OOM to estimate batch size."
            raise RuntimeError(msg)

        def distance_from_bound(val: tuple[int, int | float]) -> float:
            if val[1] < self._mem_lower_bound:
                # if memory usage is same, then higher batch size is preferred
                return self._mem_lower_bound - val[1] - val[0] / 10000
            if self._mem_upper_bound < val[1]:
                # if memory usage is same, then lower batch size is preferred
                return val[1] - self._mem_upper_bound + val[0] / 10000
            return min(abs(self._mem_lower_bound - val[1]), abs(val[1] - self._mem_upper_bound))

        bs_arr = sorted(self._bs_try_history.items(), key=lambda x: x[0])
        for idx in range(len(bs_arr) - 1, -1, -1):
            if bs_arr[idx][1] < self._mem_upper_bound:
                cur_max_bs_idx = idx
                break
        else:
            logger.warning("All batch size tried used more memory size than upper bound.")
            return bs_arr[0][0]

        def check_bs_suitable(estimated_bs: int) -> bool:
            # Check batch size is between largest bs which uses lower memory than uppper bound
            # and smallest bs which uses higher memory than upper bound.
            if estimated_bs >= bs_arr[cur_max_bs_idx][0]:
                if cur_max_bs_idx + 1 < len(bs_arr):
                    if estimated_bs < bs_arr[cur_max_bs_idx + 1][0]:
                        return True
                else:
                    return True
            return False

        x_idx, y_idx = 0, len(bs_arr) - 1

        while x_idx < y_idx:
            graident = (bs_arr[y_idx][1] - bs_arr[x_idx][1]) / (bs_arr[y_idx][0] - bs_arr[x_idx][0])
            b = bs_arr[y_idx][1] - graident * bs_arr[y_idx][0]
            if graident != 0:
                estimated_bs = round(((self._total_mem * estimation_pct) - b) / (graident * 2)) * 2
                if check_bs_suitable(estimated_bs):
                    break

            if distance_from_bound(bs_arr[x_idx + 1]) < distance_from_bound(bs_arr[y_idx - 1]):
                x_idx += 1
            else:
                y_idx -= 1

        if x_idx == y_idx:
            if check_bs_suitable(bs_arr[cur_max_bs_idx][0] + 2):
                estimated_bs = bs_arr[cur_max_bs_idx][0] + 2
            else:
                estimated_bs = bs_arr[cur_max_bs_idx][0]

        if estimated_bs > self._max_bs:
            estimated_bs = self._max_bs

        return estimated_bs


def _run_trial(train_func: Callable[[int], Any], bs: int, trial_queue: mp.Queue) -> None:
    mp.set_start_method(None, True)  # reset mp start method

    oom = False
    try:
        train_func(bs)
    except RuntimeError as e:
        if str(e).startswith("CUDA out of memory.") or str(e).startswith(  # CUDA OOM
            "Allocation is out of device memory on current platform.",  # XPU OOM
        ):
            oom = True
        else:
            raise
    except AttributeError as e:
        if str(e).startswith("'NoneType' object has no attribute 'best_model_path'"):
            pass
        else:
            raise

    trial_queue.put(
        {
            "oom": oom,
            "max_memory_reserved": _get_max_memory_reserved(),
        },
    )


def _get_max_memory_reserved() -> int:
    if is_xpu_available():
        return torch.xpu.max_memory_reserved(device=None)
    return torch.cuda.max_memory_reserved(device=None)


def _get_total_memory_size() -> int:
    if is_xpu_available():
        return torch.xpu.get_device_properties(0).total_memory
    _, total_mem = torch.cuda.mem_get_info()
    return total_mem
