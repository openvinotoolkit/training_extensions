"""Algorithm to find a proper batch size which is fit to current GPU device."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Callable

import torch

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


def adapt_batch_size(train_func: Callable[[int], None], default_bs: int, trainset_size: int) -> int:
    """Decrease batch size if default batch size isn't fit to current GPU device.

    Args:
        train_func (Callable[[int], None]): training function with single arugment to set batch size
        default_bs (int): default batch size
        trainset_size (int): training dataset size

    Returns:
        int: proper batch size possibly decreased as default value isn't fit
    """
    if default_bs <= 0:
        raise ValueError("Batch size should be bigger than 0.")
    if trainset_size <= 0:
        raise ValueError("train data set size should be bigger than 0.")

    if trainset_size < default_bs:
            default_bs = trainset_size

    min_bs = 0
    lowest_unavailable_bs = default_bs + 2
    _, total_mem = torch.cuda.mem_get_info()

    def _get_even_center_val(val1, val2):
        ret = (val1 + val2) // 2
        if ret % 2 == 1:
            ret += 1
        return ret

    while True:
        cuda_oom = False
        torch.cuda.reset_max_memory_allocated(device=None)

        try:
            train_func(default_bs)
        except RuntimeError as e:
            if str(e).startswith('CUDA out of memory.'):
                print(str(e))
                cuda_oom = True
            else:
                raise e

        gpu_memory_usage = torch.cuda.max_memory_allocated(device=None) / total_mem
        logger.debug(
            f"Adapting Batch size => bs : {default_bs}, CUDA_OOM : {cuda_oom}, GPU memory usage : {gpu_memory_usage}%"
        )

        # If GPU memory usage is too close to limit, CUDA OOM can be raised during training
        if cuda_oom or torch.cuda.max_memory_allocated(device=None) / total_mem >= 0.87:
            if default_bs < lowest_unavailable_bs:
                lowest_unavailable_bs = default_bs
            default_bs = _get_even_center_val(default_bs, min_bs)
        else:
            min_bs = default_bs
            default_bs = _get_even_center_val(default_bs, lowest_unavailable_bs)

        torch.cuda.empty_cache()
        if lowest_unavailable_bs is not None and lowest_unavailable_bs - min_bs <= 2:
            break

    if min_bs == 0:
        raise RuntimeError("Current device can't train model even with 2!")
    
    return min_bs
