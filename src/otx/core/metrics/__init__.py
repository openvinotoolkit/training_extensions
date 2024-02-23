# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX custom metrices."""

from typing import Callable

from torchmetrics import Metric

from .accuracy import CustomAccuracy, HLabelAccuracy

MetricCallable = Callable[[], Metric]

__all__ = ["CustomAccuracy", "HLabelAccuracy"]
