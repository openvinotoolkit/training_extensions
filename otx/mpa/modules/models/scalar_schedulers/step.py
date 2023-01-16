# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

from ..builder import SCALAR_SCHEDULERS
from .base import BaseScalarScheduler


@SCALAR_SCHEDULERS.register_module()
class StepScalarScheduler(BaseScalarScheduler):
    def __init__(self, scales, num_iters, by_epoch=False):
        super(StepScalarScheduler, self).__init__()

        self.by_epoch = by_epoch

        assert len(scales) == len(num_iters) + 1
        assert len(scales) > 0

        self._scales = list(scales)
        self._iter_ranges = list(num_iters) + [np.iinfo(np.int32).max]

    def _get_value(self, step, epoch_size):
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
