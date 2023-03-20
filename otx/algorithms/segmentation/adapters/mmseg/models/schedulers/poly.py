"""Polynomial scheduler."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

from otx.algorithms.segmentation.adapters.mmseg.utils.builder import SCALAR_SCHEDULERS

from .base import BaseScalarScheduler


@SCALAR_SCHEDULERS.register_module()
class PolyScalarScheduler(BaseScalarScheduler):
    """The learning rate changes over time according to a polynomial schedule.

    Args:
        start_scale (float): The initial learning rate scale.
        end_scale (float): The final learning rate scale.
        num_iters (int): The number of iterations to reach the final learning rate.
        power (float): The power of the polynomial schedule.
        by_epoch (bool): Whether to use epoch as the unit of iteration.
    """

    def __init__(
        self, start_scale: float, end_scale: float, num_iters: int, power: float = 1.2, by_epoch: bool = False
    ):
        super().__init__()

        self._start_s = start_scale
        assert self._start_s >= 0.0
        self._end_s = end_scale
        assert self._end_s >= 0.0
        self._num_iters = num_iters
        assert self._num_iters >= 0
        self._power = power
        assert self._power >= 0.0
        self.by_epoch = by_epoch

    def _get_value(self, step, epoch_size):
        if step is None:
            return float(self._end_s)

        if self.by_epoch:
            num_iters = epoch_size * self._num_iters
        else:
            num_iters = self._num_iters
        if num_iters == 0:
            return self._end_s

        if step < num_iters:
            factor = (self._end_s - self._start_s) / (1.0 - self._power)
            var_a = factor / (num_iters**self._power)
            var_b = -factor * self._power / float(num_iters)

            out_value = var_a * np.power(step, self._power) + var_b * step + self._start_s
        else:
            out_value = self._end_s

        return out_value
