"""Unbiased-teacher hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS

from otx.algorithms.common.adapters.mmcv.hooks.dual_model_ema_hook import (
    DualModelEMAHook,
)
from otx.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class MeanTeacherHook(DualModelEMAHook):
    """MeanTeacherHook for semi-supervised learnings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unlabeled_loss_enabled = False

    def before_train_epoch(self, runner):
        """Enable unlabeled loss if over start epoch."""
        if runner.epoch + 1 < self.start_epoch:
            return
        if self.unlabeled_loss_enabled:
            return

        super().before_train_epoch(runner)

        average_pseudo_label_ratio = self._get_average_pseudo_label_ratio(runner)
        logger.info(f"avr_ps_ratio: {average_pseudo_label_ratio}")
        self._get_model(runner).enable_unlabeled_loss(True)
        self.unlabeled_loss_enabled = True
        logger.info("---------- Enabled unlabeled loss and EMA smoothing ----------")

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if runner.iter % self.interval != 0:
            # Skip update
            return

        if runner.epoch + 1 < self.start_epoch or self.unlabeled_loss_enabled is False:
            # Just copy parameters before enabled
            self._copy_model()
            return

        # EMA
        self._ema_model()

    def _get_average_pseudo_label_ratio(self, runner):
        output_backup = runner.log_buffer.output.copy()
        was_ready = runner.log_buffer.ready
        runner.log_buffer.average(100)
        average_pseudo_label_ratio = runner.log_buffer.output.get("ps_ratio", 0.0)
        runner.log_buffer.output = output_backup
        runner.ready = was_ready
        return average_pseudo_label_ratio
