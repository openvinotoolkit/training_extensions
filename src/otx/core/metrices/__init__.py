# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX custom metrices."""

from typing import Callable, Union

from torchmetrics import Metric

MetricCallable = Union[Callable[[], Metric], Callable[[int], Metric]]

__all__ = ["HLabelAccuracy"]
