"""Module for defining hook for semi-supervised learning for classification task."""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
from typing import Any

from lightning import Callback, LightningModule, Trainer


class UnlabeledLossWarmUpCallback(Callback):
    """Hook for SemiSL for classification.

    This hook includes unlabeled warm-up loss coefficient (default: True):
        unlabeled_coef = (0.5 - cos(min(pi, 2 * pi * k) / K)) / 2
        k: current step, K: total steps

    Args:
        warmup_steps_ratio (float): Ratio of warm-up steps to total steps (default: 0.2).
    """

    def __init__(self, warmup_steps_ratio: float = 0.2):
        self.warmup_steps_ratio = warmup_steps_ratio
        self.total_steps = 0
        self.current_step, self.unlabeled_coef = 0, 0.0
        self.num_pseudo_label = 0

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
    ) -> None:
        """Calculate the unlabeled warm-up loss coefficient before training iteration."""
        if self.unlabeled_coef < 1.0:
            if self.total_steps == 0:
                dataloader = (
                    trainer.train_dataloader["labeled"]
                    if isinstance(trainer.train_dataloader, dict)
                    else trainer.train_dataloader
                )
                self.total_steps = int(trainer.max_epochs * len(dataloader) * self.warmup_steps_ratio)
            self.unlabeled_coef = 0.5 * (
                1 - math.cos(min(math.pi, (2 * math.pi * self.current_step) / self.total_steps))
            )
            if trainer.model is None:
                msg = "Model is not found in the trainer."
                raise ValueError(msg)
            trainer.model.model.unlabeled_coef = self.unlabeled_coef
        self.current_step += 1
