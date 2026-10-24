# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-epoch summary logger.

Lightning's ``RichProgressBar`` is configured with ``leave=False`` (see
``configure_callbacks`` in ``backend/lightning/engine.py``), so the in-place
progress display is cleared at the end of each epoch. This is desirable
interactively, but means a captured log file (``nohup``, CI artifacts, ...)
ends up with only the final epoch's snapshot.

This callback emits a single, plain-text, line-buffered summary at the end of
every validation epoch, so logs accumulate one entry per epoch with the
metrics that matter (loss, monitored metric, LR). It is cheap, has no
interactive effect (RichProgressBar runs on a different stream/transient
region), and survives stdout redirection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from lightning import Callback

if TYPE_CHECKING:
    import lightning.pytorch as pl

logger = logging.getLogger(__name__)


def _scalar(value: object) -> float | None:
    """Best-effort conversion of a Lightning callback metric to ``float``."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class EpochSummary(Callback):
    """Log a one-line summary of training/validation metrics per epoch.

    The summary is emitted via the standard ``logging`` module at INFO level,
    so it is captured by any handler the engine has configured (file logging,
    stdout, etc.) and timestamps line up with the rest of the training log.

    Args:
        keys: Metric keys to include in the summary, looked up in
            ``trainer.callback_metrics``. Missing keys are silently skipped so
            this callback works for every task without per-task wiring.
        decimals: Number of decimal digits used when formatting metric values.
    """

    # Ordered, task-agnostic shortlist. The first matching key per task is
    # picked up; the others are skipped silently. New tasks can extend this
    # list without breaking existing ones.
    DEFAULT_KEYS: tuple[str, ...] = (
        "train/total_loss",
        "train/loss",
        "val/PCK",
        "val/accuracy",
        "val/Dice",
        "val/mIoU",
        "val/map_50",
        "val/map",
        "val/f1-score",
        "val/image_AUROC",
        "lr",
    )

    def __init__(
        self,
        keys: tuple[str, ...] | list[str] | None = None,
        decimals: int = 4,
    ) -> None:
        super().__init__()
        self._keys = tuple(keys) if keys is not None else self.DEFAULT_KEYS
        self._decimals = int(decimals)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Emit the per-epoch summary after validation has finished."""
        # Skip Lightning's sanity-check pass at the start of training; those
        # metrics are noisy and would offset the displayed epoch counter.
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        parts: list[str] = []
        for key in self._keys:
            value = _scalar(metrics.get(key))
            if value is None:
                continue
            parts.append(f"{key}={value:.{self._decimals}f}")

        # Always show the epoch counter even when no whitelisted metric is
        # present (e.g. very first epoch with only train metrics flushed).
        epoch = trainer.current_epoch + 1  # Epochs are 0 indexed, add one to clearly log current epoch
        max_epochs = trainer.max_epochs
        header = f"epoch {epoch:>3}/{max_epochs}" if max_epochs is not None else f"epoch {epoch:>3}"

        if parts:
            logger.info("%s | %s", header, "  ".join(parts))
        else:
            logger.info("%s | (no whitelisted metrics)", header)
