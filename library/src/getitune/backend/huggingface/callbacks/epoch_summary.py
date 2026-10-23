# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-epoch summary logger for the Hugging Face backend.

``transformers.Trainer`` only reports training progress through its tqdm
``ProgressCallback`` (terminal output), and the engine disables every
experiment-tracker integration (``report_to=[]``). Nothing reaches the
standard ``logging`` hierarchy during the training loop, so captured logs
(e.g. application job logs) stay silent until training ends.

This callback mirrors the Lightning backend's ``EpochSummary`` and logs one
line per epoch with the mean training loss, the validation metrics, and the
learning rate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from transformers import TrainerCallback

from getitune.backend.huggingface.trainers.utils import remap_log_key
from getitune.utils.epoch_summary import DEFAULT_EPOCH_SUMMARY_KEYS, format_epoch_summary, to_scalar

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments

__all__ = ["HFEpochSummaryCallback"]

logger = logging.getLogger(__name__)

_TRAIN_LOSS_KEY = "train/total_loss"


class HFEpochSummaryCallback(TrainerCallback):
    """Log a one-line summary of training/validation metrics per epoch.

    ``Trainer`` evaluates *after* ``on_epoch_end``, so the summary is emitted
    from ``on_evaluate`` when validation runs for the epoch. Epochs without
    validation (no val subset, or skipped by ``val_check_interval``) are
    flushed at the start of the next epoch or at the end of training.

    Args:
        keys: Metric keys (after ``train/`` / ``val/`` remapping) to include.
            Missing keys are silently skipped.
        decimals: Number of decimal digits used when formatting metric values.
    """

    def __init__(self, keys: tuple[str, ...] | list[str] | None = None, decimals: int = 4) -> None:
        self._keys = tuple(keys) if keys is not None else DEFAULT_EPOCH_SUMMARY_KEYS
        self._decimals = int(decimals)
        self._metrics: dict[str, float] = {}
        self._loss_sum = 0.0
        self._loss_count = 0
        self._pending_epoch: int | None = None

    def _reset(self) -> None:
        self._metrics = {}
        self._loss_sum = 0.0
        self._loss_count = 0
        self._pending_epoch = None

    def _flush(self, state: TrainerState) -> None:
        if self._pending_epoch is None:
            return
        if state.is_world_process_zero:
            metrics = dict(self._metrics)
            if self._loss_count:
                metrics[_TRAIN_LOSS_KEY] = self._loss_sum / self._loss_count
            max_epochs = int(state.num_train_epochs) if state.num_train_epochs else None
            logger.info(
                "%s",
                format_epoch_summary(
                    epoch=self._pending_epoch,
                    max_epochs=max_epochs,
                    metrics=metrics,
                    keys=self._keys,
                    decimals=self._decimals,
                ),
            )
        self._reset()

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Flush the previous epoch if it finished without validation."""
        self._flush(state)
        self._reset()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Collect step losses (averaged per epoch) and the latest value of every other metric."""
        for key, value in (logs or {}).items():
            scalar = to_scalar(value)
            if scalar is None:
                continue
            name = remap_log_key(key)
            if name == _TRAIN_LOSS_KEY:
                self._loss_sum += scalar
                self._loss_count += 1
            else:
                self._metrics[name] = scalar

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Mark the epoch as finished; the summary waits for validation, if any."""
        self._pending_epoch = round(state.epoch) if state.epoch is not None else None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Emit the summary once validation for the finished epoch is available."""
        for key, value in (metrics or {}).items():
            scalar = to_scalar(value)
            if scalar is not None:
                self._metrics[remap_log_key(key)] = scalar
        self._flush(state)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Flush the last epoch if it finished without validation."""
        self._flush(state)
