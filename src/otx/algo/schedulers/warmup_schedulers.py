# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Warm-up schedulers for the OTX2.0."""
from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch.cli import ReduceLROnPlateau

if TYPE_CHECKING:
    from torch.optim import Optimizer


class BaseWarmupScheduler:
    """Base Warumup Scheduler class.

    It should be inherited if want to implement warmup based Custom LRScheduler.
    i.e. WarmupCustomLRScheduler(BaseWarmupScheduler, ...), WarmupCosineAnnealingLR(BaseWarmupScheduler, ...)

    Args:
        warmup_steps (int): The total number of the warmup steps. it could be epoch or iter.
        warmup_by_epoch (bool): If True, warmup_steps represent the epoch.

    """

    warmup_steps: int
    warmup_by_epoch: bool


class WarmupReduceLROnPlateau(BaseWarmupScheduler, ReduceLROnPlateau):
    """ReduceLROnPlateau for enabling the warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): The total number of the warmup steps. it could be epoch or iter.
        monitor (str): The name of monitoring value.
        warmup_by_epoch (bool): If True, warmup_steps represent the epoch.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        monitor: str,
        warmup_by_epoch: bool = False,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        verbose: bool = False,
    ):
        self.warmup_steps = warmup_steps
        self.warmup_by_epoch = warmup_by_epoch
        super().__init__(
            optimizer,
            monitor,
            mode,
            factor,
            patience,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
            verbose,
        )
