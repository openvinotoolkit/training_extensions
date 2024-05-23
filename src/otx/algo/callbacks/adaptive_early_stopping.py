# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Callback for early stopping with warmup possibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

if TYPE_CHECKING:
    import lightning.pytorch as pl


class EarlyStoppingWithWarmup(EarlyStopping):
    """EarlyStoppingWithWarmup callback."""

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: float | None = None,
        divergence_threshold: float | None = None,
        check_on_train_epoch_end: bool | None = None,
        log_rank_zero_only: bool = False,
        warmup_iters: int = 100,
        warmup_epochs: int = 3,
    ):
        """EarlyStoppingWithWarmup callback.

        Args:
            monitor (str): The metric to monitor.
            min_delta (float, optional): Minimum change in the monitored quantity
                to qualify as an improvement. Defaults to 0.0.
            patience (int, optional): Number of epochs with no improvement
                after which training will be stopped. Defaults to 3.
            verbose (bool, optional): If True, prints messages to stdout. Defaults to False.
            mode (str, optional): One of {"min", "max"}. In "min" mode, training will stop when
                the quantity monitored has stopped decreasing. In "max" mode,
                it will stop when the quantity monitored has stopped increasing. Defaults to "min".
            strict (bool, optional): If True, the monitored quantity must improve
                according to the mode for it to be considered an improvement.
                Defaults to True.
            check_finite (bool, optional): If True, check that the monitored quantity is
                finite before considering an improvement. Defaults to True.
            stopping_threshold (float | None, optional): The threshold to stop training.
                Defaults to None.
            divergence_threshold (float | None, optional): The threshold for divergence detection.
                Defaults to None.
            check_on_train_epoch_end (bool | None, optional): If True,
                checks the stopping criterion on train_epoch_end. Defaults to None.
            log_rank_zero_only (bool, optional): If True, logs should only be printed from rank 0.
                Defaults to False.
            warmup_iters (int, optional): Number of warmup iterations. Defaults to 100.
            warmup_epochs (int, optional): Number of warmup epochs. Defaults to 3.
        """
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
            log_rank_zero_only=log_rank_zero_only,
        )
        # two thresholds to have invariant to extra small datasets and larger datasets
        self.warmup_iters = warmup_iters
        self.warmup_epochs = warmup_epochs

    def _should_skip_check(self, trainer: pl.Trainer) -> bool:
        warmup_threshold = max(self.warmup_epochs * trainer.num_training_batches, self.warmup_iters)
        return super()._should_skip_check(trainer) or trainer.global_step < warmup_threshold
