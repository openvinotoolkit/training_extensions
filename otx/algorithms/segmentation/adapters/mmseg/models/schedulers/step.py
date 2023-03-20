"""Step scheduler."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List

import numpy as np

from otx.algorithms.segmentation.adapters.mmseg.utils.builder import SCALAR_SCHEDULERS

from .base import BaseScalarScheduler


@SCALAR_SCHEDULERS.register_module()
class StepScalarScheduler(BaseScalarScheduler):
    """Step learning rate scheduler.

    Example:
        >>> scheduler = StepScalarScheduler(scales=[1.0, 0.1, 0.01], num_iters=[100, 200])
        This means that the learning rate will be 1.0 for the first 100 iterations,
        0.1 for the next 200 iterations, and 0.01 for the rest of the iterations.

    Args:
        scales (List[int]): List of learning rate scales.
        num_iters (List[int]): A list specifying the count of iterations at each scale.
        by_epoch (bool): Whether to use epoch as the unit of iteration.
    """

    def __init__(self, scales: List[float], num_iters: List[int], by_epoch: bool = False):
        super().__init__()

        self.by_epoch = by_epoch

        assert len(scales) == len(num_iters) + 1
        assert len(scales) > 0

        self._scales = list(scales)
        self._iter_ranges = list(num_iters) + [np.iinfo(np.int32).max]

    def _get_value(self, step, epoch_size) -> float:
        if step is None:
            return float(self._scales[-1])

        out_scale_idx = 0
        for iter_range in self._iter_ranges:
            if self.by_epoch:
                iter_threshold = epoch_size * iter_range
            else:
                iter_threshold = iter_range

            if step < iter_threshold:
                break

            out_scale_idx += 1

        return float(self._scales[out_scale_idx])
