# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Warm-up schedulers for the OTX2.0."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from otx.core.schedulers.callable import SchedulerCallableSupportHPO

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, ReduceLROnPlateau
    from torch.optim.optimizer import Optimizer


class LinearWarmupScheduler(LambdaLR):
    """Linear Warmup scheduler.

    Args:
        num_warmup_steps: Learning rate will linearly increased during the period same as this number.
        warmup_interval: If "epoch", count the number of steps for the warmup period.
            Otherwise, the iteration step will be the warmup period.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int = 1000,
        interval: Literal["step", "epoch"] = "step",
    ):
        if not num_warmup_steps > 0:
            msg = f"num_warmup_steps should be > 0, got {num_warmup_steps}"
            raise ValueError(msg)
        self.num_warmup_steps = num_warmup_steps
        self.interval = interval
        super().__init__(optimizer, lambda step: min((step + 1.0) / self.num_warmup_steps, 1.0))

    def step(self, epoch: int | None = None) -> None:
        """Overriding the step to disable the warmup scheduler after n_steps."""
        if self.activated:
            super().step(epoch)

    @property
    def activated(self) -> bool:
        """If true, the current step count is less than the num_warmup_steps."""
        return self._step_count <= self.num_warmup_steps


class LinearWarmupSchedulerCallable:
    """This callable can create the given main LR scheduler and `LinearWarmupScheduler` at the same time.

    Args:
        main_scheduler_callable: Callable to create a LR scheduler that will be mainly used.
        num_warmup_steps: Learning rate will linearly increased during the period same as this number.
            If it is less than equal to zero, do not create `LinearWarmupScheduler`.
        warmup_interval: If "epoch", count the number of steps for the warmup period.
            Otherwise, the iteration step will be the warmup period.
        monitor: If given, override the main scheduler's `monitor` attribute.
    """

    def __init__(
        self,
        main_scheduler_callable: LRSchedulerCallable,
        num_warmup_steps: int = 0,
        warmup_interval: Literal["step", "epoch"] = "step",
        monitor: str | None = None,
    ):
        self.main_scheduler_callable = SchedulerCallableSupportHPO.from_callable(main_scheduler_callable)
        self.num_warmup_steps = num_warmup_steps
        self.warmup_interval = warmup_interval
        self.monitor = monitor

    def __call__(self, optimizer: Optimizer) -> list[LRScheduler | ReduceLROnPlateau]:
        """Create a list of lr schedulers."""
        main_scheduler = self.main_scheduler_callable(optimizer)

        if self.monitor and hasattr(main_scheduler, "monitor"):
            main_scheduler.monitor = self.monitor

        schedulers = [main_scheduler]

        if self.num_warmup_steps > 0:
            schedulers += [
                LinearWarmupScheduler(
                    optimizer=optimizer,
                    num_warmup_steps=self.num_warmup_steps,
                    interval=self.warmup_interval,
                ),
            ]

        return schedulers
