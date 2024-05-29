# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Timer for logging iteration time for train, val, and test phases."""

from __future__ import annotations

from collections import defaultdict
from time import time
from typing import TYPE_CHECKING, Any

from lightning import Callback, LightningModule, Trainer

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT


class IterationTimer(Callback):
    """Timer for logging iteration time for train, val, and test phases."""

    def __init__(
        self,
        prog_bar: bool = True,
        on_step: bool = True,
        on_epoch: bool = True,
    ) -> None:
        super().__init__()
        self.prog_bar = prog_bar
        self.on_step = on_step
        self.on_epoch = on_epoch

        self.start_time: dict[str, float] = defaultdict(float)
        self.end_time: dict[str, float] = defaultdict(float)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset timer before every train epoch starts."""
        self.start_time.clear()
        self.end_time.clear()

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset timer before every validation epoch starts."""
        self.start_time.clear()
        self.end_time.clear()

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset timer before every test epoch starts."""
        self.start_time.clear()
        self.end_time.clear()

    def _on_batch_start(
        self,
        pl_module: LightningModule,
        phase: str,
        batch_size: int,
    ) -> None:
        self.start_time[phase] = time()

        if not self.end_time[phase]:
            return

        name = f"{phase}/data_time"

        data_time = self.start_time[phase] - self.end_time[phase]

        pl_module.log(
            name=name,
            value=data_time,
            prog_bar=self.prog_bar,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
            batch_size=batch_size,
        )

    def _on_batch_end(
        self,
        pl_module: LightningModule,
        phase: str,
        batch_size: int,
    ) -> None:
        if not self.end_time[phase]:
            self.end_time[phase] = time()
            return

        name = f"{phase}/iter_time"
        curr_end_time = time()
        iter_time = curr_end_time - self.end_time[phase]
        self.end_time[phase] = curr_end_time

        pl_module.log(
            name=name,
            value=iter_time,
            prog_bar=self.prog_bar,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
            batch_size=batch_size,
        )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
    ) -> None:
        """Log iteration data time on the training batch start."""
        if isinstance(batch, dict):
            batch_size = 0
            for key in batch:
                batch_size += batch[key].batch_size
        else:
            batch_size = batch.batch_size
        self._on_batch_start(
            pl_module=pl_module,
            phase="train",
            batch_size=batch_size,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
    ) -> None:
        """Log iteration time on the training batch end."""
        if isinstance(batch, dict):
            batch_size = 0
            for key in batch:
                batch_size += batch[key].batch_size
        else:
            batch_size = batch.batch_size
        self._on_batch_end(
            pl_module=pl_module,
            phase="train",
            batch_size=batch_size,
        )

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log iteration data time on the validation batch start."""
        self._on_batch_start(
            pl_module=pl_module,
            phase="validation",
            batch_size=batch.batch_size,
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log iteration time on the validation batch end."""
        self._on_batch_end(
            pl_module=pl_module,
            phase="validation",
            batch_size=batch.batch_size,
        )

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log iteration data time on the test batch start."""
        self._on_batch_start(
            pl_module=pl_module,
            phase="test",
            batch_size=batch.batch_size,
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log iteration time on the test batch end."""
        self._on_batch_end(
            pl_module=pl_module,
            phase="test",
            batch_size=batch.batch_size,
        )
