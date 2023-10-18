"""Utils for computation of repeat times."""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

from otx.v2.api.utils.logger import get_logger

logger = get_logger()


def get_proper_repeat_times(
    data_size: int,
    batch_size: int,
    coef: float,
    min_repeat: float,
) -> float:
    """Get proper repeat times for adaptive training.

    Args:
        data_size (int): The total number of the training dataset
        batch_size (int): The batch size for the training data loader
        coef (float) : coefficient that effects to number of repeats
                       (coef * math.sqrt(num_iters-1)) +5
        min_repeat (float) : minimum repeats
    """
    if data_size == 0 or batch_size == 0:
        logger.info("Repeat dataset enabled, but not a train mode. repeat times set to 1.")
        return 1
    n_iters_per_epoch = math.ceil(data_size / batch_size)
    return math.floor(max(coef * math.sqrt(n_iters_per_epoch - 1) + 5, min_repeat))
