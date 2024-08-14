# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for exponential moving average for SemiSL mean teacher algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning import Callback, LightningModule, Trainer

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT


class EMAMeanTeacher(Callback):
    """callback for SemiSL MeanTeacher algorithm.

    This callback averages the weights of the teacher model.

    Args:
        momentum (float, optional): momentum. Defaults to 0.999.
        start_epoch (int, optional): start epoch. Defaults to 1.
    """

    def __init__(
        self,
        momentum: float = 0.999,
        start_epoch: int = 1,
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.start_epoch = start_epoch
        self.synced_models = False

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Set up src & dst model parameters."""
        # call to nn.model
        model = trainer.model.model
        self.src_model = getattr(model, "student_model", None)
        self.dst_model = getattr(model, "teacher_model", None)
        if self.src_model is None or self.dst_model is None:
            msg = "student_model and teacher_model should be set for MeanTeacher algorithm"
            raise RuntimeError(msg)
        self.src_params = self.src_model.state_dict(keep_vars=True)
        self.dst_params = self.dst_model.state_dict(keep_vars=True)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
    ) -> None:
        """Update ema parameter every iteration."""
        if trainer.current_epoch < self.start_epoch:
            return

        # EMA
        self._ema_model(trainer.global_step)

    def _copy_model(self) -> None:
        with torch.no_grad():
            for name, src_param in self.src_params.items():
                if src_param.requires_grad:
                    dst_param = self.dst_params[name]
                    dst_param.data.copy_(src_param.data)

    def _ema_model(self, global_step: int) -> None:
        if self.start_epoch != 0 and not self.synced_models:
            self._copy_model()
            self.synced_models = True

        momentum = min(1 - 1 / (global_step + 1), self.momentum)
        with torch.no_grad():
            for name, src_param in self.src_params.items():
                if src_param.requires_grad:
                    dst_param = self.dst_params[name]
                    dst_param.data.copy_(dst_param.data * momentum + src_param.data * (1 - momentum))
