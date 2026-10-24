# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom schedulers for the getitune2.0."""

from __future__ import annotations

from typing import Callable

from lightning.pytorch.cli import ReduceLROnPlateau
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from getitune.backend.lightning.schedulers.callable import SchedulerCallableSupportAdaptiveBS
from getitune.backend.lightning.schedulers.warmup_schedulers import LinearWarmupScheduler, LinearWarmupSchedulerCallable

LRSchedulerListCallable = Callable[[Optimizer], list[LRScheduler | ReduceLROnPlateau]]

__all__ = [
    "LRSchedulerListCallable",
    "LinearWarmupScheduler",
    "LinearWarmupSchedulerCallable",
    "SchedulerCallableSupportAdaptiveBS",
]
