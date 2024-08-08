# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom schedulers for the OTX2.0."""

from __future__ import annotations

from typing import Callable

from lightning.pytorch.cli import ReduceLROnPlateau
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from otx.core.schedulers.callable import SchedulerCallableSupportHPO
from otx.core.schedulers.warmup_schedulers import LinearWarmupScheduler, LinearWarmupSchedulerCallable

LRSchedulerListCallable = Callable[[Optimizer], list[LRScheduler | ReduceLROnPlateau]]

__all__ = [
    "LRSchedulerListCallable",
    "LinearWarmupScheduler",
    "LinearWarmupSchedulerCallable",
    "SchedulerCallableSupportHPO",
]
