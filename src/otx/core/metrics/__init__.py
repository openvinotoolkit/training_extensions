# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX custom metrices."""

from typing import Callable

from lightning.pytorch.cli import ArgsType
from torchmetrics import Metric

MetricCallable = Callable[[ArgsType], Metric]

from .accuracy import CustomAccuracy, HLabelAccuracy

__all__ = ["CustomAccuracy", "HLabelAccuracy"]