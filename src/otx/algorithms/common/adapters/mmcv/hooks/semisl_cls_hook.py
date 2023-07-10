"""Module for defining hook for semi-supervised learning for classification task."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SemiSLClsHook(Hook):
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

    def __init__(self, total_steps=0, unlabeled_warmup=True):
        self.unlabeled_warmup = unlabeled_warmup
        self.total_steps = total_steps
        self.current_step, self.unlabeled_coef = 0, 0
        self.num_pseudo_label = 0

    def before_train_iter(self, runner):
        """Calculate the unlabeled warm-up loss coefficient before training iteration."""
        if self.unlabeled_warmup and self.unlabeled_coef < 1.0:
            if self.total_steps == 0:
                self.total_steps = runner.max_iters
            self.unlabeled_coef = 0.5 * (
                1 - math.cos(min(math.pi, (2 * math.pi * self.current_step) / self.total_steps))
            )
            model = self._get_model(runner)
            model.head.unlabeled_coef = self.unlabeled_coef
        self.current_step += 1

    def after_train_iter(self, runner):
        """Add the number of pseudo-labels correctly selected from iteration."""
        model = self._get_model(runner)
        self.num_pseudo_label += int(model.head.num_pseudo_label)

    def after_epoch(self, runner):
        """Add data related to Semi-SL to the log."""
        if self.unlabeled_warmup:
            runner.log_buffer.output.update({"unlabeled_coef": round(self.unlabeled_coef, 4)})
        runner.log_buffer.output.update({"pseudo_label": self.num_pseudo_label})
        self.num_pseudo_label = 0

    def _get_model(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        return model
