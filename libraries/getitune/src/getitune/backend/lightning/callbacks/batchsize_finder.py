"""Callback that finds the optimal batch size."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.logger import DummyLogger

from getitune.utils.device import is_xpu_available

if TYPE_CHECKING:
    from lightning import LightningModule
    from lightning.pytorch.trainer import Trainer


class BatchSizeFinder(Callback):
    """This callback makes trainer run specified iteration and exit.

    Args:
        steps_per_trial: number of steps to run with a given batch size.
            Ideally 1 should be enough to test if an OOM error occurs, however in practice a few are needed.
    """

    def __init__(
        self,
        steps_per_trial: int = 5,
    ) -> None:
        self._steps_per_trial = steps_per_trial

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None) -> None:
        """Check current stage is fit."""
        if stage != "fit":
            msg = "Adaptive batch size supports only training."
            raise RuntimeError(msg)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run steps_per_trial iterations and exit."""
        _scale_batch_reset_params(trainer, self._steps_per_trial)
        _try_loop_run(trainer)


def _try_loop_run(trainer: Trainer) -> None:
    loop = trainer._active_loop  # noqa: SLF001
    if loop is None:
        msg = "There is no active loop."
        raise RuntimeError(msg)
    loop.restarting = False
    loop.run()


def _scale_batch_reset_params(trainer: Trainer, steps_per_trial: int, max_epochs: int = 1) -> None:
    trainer.logger = DummyLogger() if trainer.logger is not None else None
    trainer.callbacks = []
    # For XPU devices 1 epoch sometimes is not enough to catch an error.
    # Emperically enlarge this to 15 iterations (steps_per_trial * epochs)
    max_epochs = 3 if is_xpu_available() else 1

    loop = trainer._active_loop  # noqa: SLF001
    if loop is None:
        msg = "There is no active loop."
        raise RuntimeError(msg)
    if trainer.fit_loop.epoch_loop.max_steps == -1:  # epoch based loop
        trainer.fit_loop.max_epochs = max_epochs
        trainer.limit_train_batches = steps_per_trial
    else:  # iter based loop
        trainer.fit_loop.epoch_loop.max_steps = steps_per_trial
        trainer.limit_train_batches = 1.0
    if trainer.limit_val_batches != 0:
        trainer.limit_val_batches = steps_per_trial
