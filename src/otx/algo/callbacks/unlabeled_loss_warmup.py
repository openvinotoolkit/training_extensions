"""Module for defining hook for semi-supervised learning for classification task."""
# Copyright (C) 2023 Intel Corporation
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
    Also, this hook adds semi-sl-related data to the log (unlabeled_coef, pseudo_label)

    Args:
        total_steps (int): total steps for training (iteration)
            Raise the coefficient from 0 to 1 during half the duration of total_steps
            default: 0, use runner.max_iters
        unlabeled_warmup (boolean): enable unlabeled warm-up loss coefficient
            If False, Semi-SL uses 1 as unlabeled loss coefficient
    """

    def __init__(self):
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
                self.total_steps = trainer.max_epochs * len(trainer.train_dataloader)
            self.unlabeled_coef = 0.5 * (
                1 - math.cos(min(math.pi, (2 * math.pi * self.current_step) / self.total_steps))
            )
            if trainer.model is None:
                msg = "Model is not found in the trainer."
                raise ValueError(msg)
            trainer.model.model.head.unlabeled_coef = self.unlabeled_coef
        self.current_step += 1
