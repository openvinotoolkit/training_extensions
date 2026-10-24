# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Simplified Learning Rate Monitor that avoids logging per-parameter-group metrics.

Models with many parameter groups (e.g., RF-DETR with layer-wise LR decay having 487+ groups)
can generate hundreds of lr-*/pg* metrics that clutter logs and callback_metrics.
This simplified version only logs a single representative LR value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from typing_extensions import override

if TYPE_CHECKING:
    import lightning.pytorch as pl


class SimpleLearningRateMonitor(LearningRateMonitor):
    """Simplified LR monitor that logs only a single representative learning rate.

    Unlike the standard LearningRateMonitor which logs lr-*/pg* for every parameter group,
    this callback logs only a single 'lr' value (from the first parameter group of the first
    optimizer/scheduler), keeping callback_metrics clean.

    Inherits from LearningRateMonitor to reuse logging interval logic and logger integration.

    Example::
        >>> from getitune.backend.lightning.callbacks.lr_monitor import SimpleLearningRateMonitor
        >>> lr_monitor = SimpleLearningRateMonitor(logging_interval='epoch')
        >>> trainer = Trainer(callbacks=[lr_monitor])
    """

    def __init__(self, logging_interval: Literal["epoch", "step"] = "epoch") -> None:
        # Don't log momentum or weight decay - keep it simple
        super().__init__(logging_interval=logging_interval, log_momentum=False, log_weight_decay=False)

    @override
    def _extract_stats(self, trainer: pl.Trainer, interval: str) -> dict[str, float]:  # type: ignore[misc]
        """Extract only a single representative learning rate instead of all param groups.

        Args:
            trainer: The Lightning trainer instance.
            interval: The logging interval ('step', 'epoch', or 'any').

        Returns:
            Dictionary with single 'lr' key containing the learning rate.
        """
        lr = self._get_single_lr(trainer)
        if lr is None:
            return {}

        latest_stat = {"lr": lr}

        # Update callback_metrics with single lr value (parent does this for all param groups)
        trainer.callback_metrics.update(
            {
                "lr": torch.tensor(lr, device=trainer.strategy.root_device),
            }
        )

        return latest_stat

    def _get_single_lr(self, trainer: pl.Trainer) -> float | None:
        """Extract learning rate from the first scheduler or optimizer."""
        # Try to get LR from scheduler configs first
        if trainer.lr_scheduler_configs:
            config = trainer.lr_scheduler_configs[0]
            opt = config.scheduler.optimizer
            if opt.param_groups:
                return opt.param_groups[0]["lr"]

        # Fall back to optimizers
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            # Handle DeepSpeed optimizer wrapper
            if hasattr(optimizer, "optimizer"):
                optimizer = optimizer.optimizer
            if optimizer.param_groups:
                return optimizer.param_groups[0]["lr"]

        return None
