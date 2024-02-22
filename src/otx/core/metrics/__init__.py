# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX custom metrices."""

from typing import Callable

from lightning.pytorch.cli import ArgsType
from torchmetrics import Metric

from .accuracy import CustomAccuracy, HLabelAccuracy

MetricCallable = Callable[[ArgsType], Metric]

__all__ = ["CustomAccuracy", "HLabelAccuracy"]
