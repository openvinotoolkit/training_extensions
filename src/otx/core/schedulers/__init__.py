# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom schedulers for the OTX2.0."""

from __future__ import annotations

from typing import Callable

from lightning.fabric.utilities.types import _TORCH_LRSCHEDULER
from lightning.pytorch.cli import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from otx.core.schedulers.warmup_schedulers import LinearWarmupScheduler, LinearWarmupSchedulerCallable

__all__ = [
    "LRSchedulerListCallable",
    "LinearWarmupScheduler",
    "LinearWarmupSchedulerCallable",
]


LRSchedulerListCallable = Callable[[Optimizer], list[_TORCH_LRSCHEDULER | ReduceLROnPlateau]]
