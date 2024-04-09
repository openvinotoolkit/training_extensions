# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom schedulers for the OTX2.0."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import dill
from lightning.fabric.utilities.types import _TORCH_LRSCHEDULER
from lightning.pytorch.cli import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from otx.core.schedulers.warmup_schedulers import LinearWarmupScheduler, LinearWarmupSchedulerCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable

__all__ = [
    "LRSchedulerListCallable",
    "LinearWarmupScheduler",
    "LinearWarmupSchedulerCallable",
]


LRSchedulerListCallable = Callable[[Optimizer], list[_TORCH_LRSCHEDULER | ReduceLROnPlateau]]


class PicklableLRSchedulerCallable:
    """It converts unpicklable lr scheduler callable such as lambda function to picklable."""

    def __init__(self, scheduler_callable: LRSchedulerCallable | LRSchedulerListCallable):
        self.dumped_scheduler_callable = dill.dumps(scheduler_callable)

    def __call__(
        self,
        optimizer: Optimizer,
    ) -> _TORCH_LRSCHEDULER | ReduceLROnPlateau | list[_TORCH_LRSCHEDULER | ReduceLROnPlateau]:
        scheduler_callable = dill.loads(self.dumped_scheduler_callable)  # noqa: S301
        return scheduler_callable(optimizer)
