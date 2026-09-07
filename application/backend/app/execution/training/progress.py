# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from lightning import Callback, LightningModule
from lightning import Trainer as LightningTrainer
from lightning.pytorch.utilities.types import STEP_OUTPUT


class TrainingProgressCallback(Callback):
    def __init__(self, on_progress_update: Callable[[float], None], min_p: float = 0, max_p: float = 100.0):
        self._on_progress_update = on_progress_update
        self._min_p = min_p
        self._max_p = max_p
        self._total_steps: int | None = None
        self._current_step: int = 0

    def _update_total_steps(self, trainer: LightningTrainer) -> None:
        if self._total_steps is not None:
            return
        max_epochs = trainer.max_epochs or 1
        steps_per_epoch = int(trainer.num_training_batches)
        self._total_steps = max(1, max_epochs * steps_per_epoch)

    def _emit_progress(self) -> None:
        if self._total_steps is None:
            return
        ratio = self._current_step / self._total_steps
        progress = self._min_p + ratio * (self._max_p - self._min_p)
        self._on_progress_update(progress)

    def on_train_batch_end(
        self,
        trainer: LightningTrainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._update_total_steps(trainer)
        self._current_step += 1
        self._emit_progress()


class EvaluationHeartbeatCallback(Callback):
    """Emit a liveness heartbeat while a Lightning test loop is running.

    The "Evaluate Model" step reports progress only when it starts and when it ends.
    A test loop that runs longer than the control plane's stale-job threshold would
    otherwise be mistaken for a hung job and terminated by a signal, killing the
    process without any Python-level error and leaving the job log truncated.

    Args:
        on_heartbeat: Called periodically to signal liveness.
        every_n_batches: Emit at most one heartbeat every N test batches, to avoid
            flooding the IPC channel on fast (e.g. GPU) evaluations.
    """

    def __init__(self, on_heartbeat: Callable[[], None], every_n_batches: int = 10) -> None:
        self._on_heartbeat = on_heartbeat
        self._every_n_batches = max(1, every_n_batches)
        self._seen = 0

    def on_test_batch_end(
        self,
        trainer: LightningTrainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._seen += 1
        if self._seen % self._every_n_batches == 0:
            self._on_heartbeat()
