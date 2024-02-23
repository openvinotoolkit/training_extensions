# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX custom metrices."""
from __future__ import annotations

from typing import Callable

from torchmetrics import Metric

from .accuracy import HLabelAccuracy

MetricCallable = Callable[[], Metric] | Callable[[int], Metric]

__all__ = ["HLabelAccuracy"]
