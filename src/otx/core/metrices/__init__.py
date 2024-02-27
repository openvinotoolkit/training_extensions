# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX custom metrices."""

from typing import Callable, Union

from torchmetrics import Metric

from .accuracy import HLabelAccuracy

MetricCallable = Union[Callable[[], Metric], Callable[[int], Metric]]

__all__ = ["HLabelAccuracy"]