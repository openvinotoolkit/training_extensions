"""Utililty classes for tracking loss dynamics."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from enum import IntEnum


class TrackingLossType(IntEnum):
    """Type of loss functions to track."""

    cls = 0
    bbox = 1
    centerness = 2
    bbox_refine = 3


class LossAccumulator:
    """Accumulate for tracking loss dynamics."""

    def __init__(self):
        self.sum = 0.0
        self.cnt = 0

    def add(self, value):
        """Add loss value to itself."""
        if isinstance(value, float):
            self.sum += value
            self.cnt += 1
        elif isinstance(value, LossAccumulator):
            self.sum += value.sum
            self.cnt += value.cnt
        else:
            raise NotImplementedError()

    @property
    def mean(self):
        """Obtain mean from the accumulated values."""
        if self.cnt == 0:
            return 0.0

        return self.sum / self.cnt
