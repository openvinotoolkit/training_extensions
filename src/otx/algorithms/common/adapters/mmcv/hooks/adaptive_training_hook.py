"""Adaptive training schedule hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math

from mmcv.runner import HOOKS, Hook, LrUpdaterHook
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.evaluation import EvalHook

from otx.algorithms.common.adapters.mmcv.hooks.early_stopping_hook import (
    EarlyStoppingHook,
)
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()

# pylint: disable=too-many-arguments, too-many-instance-attributes


@HOOKS.register_module()
class AdaptiveTrainSchedulingHook(Hook):
    """Adaptive Training Scheduling Hook.

    Depending on the size of iteration per epoch, adaptively update the validation interval.

    Args:
        max_interval (int): Maximum value of validation interval.
            Defaults to 5.
        decay (float): Parameter to control the interval. This value is set by manual manner.
            Defaults to -0.025.
        enable_adaptive_interval_hook (bool): If True, adaptive interval will be enabled.
            Defaults to False.
        enable_eval_before_run (bool): If True, initial evaluation before training will be enabled.
            Defaults to False.
    """

    def __init__(
        self,
        max_interval=5,
        decay=-0.025,
        enable_adaptive_interval_hook=False,
        enable_eval_before_run=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_interval = max_interval
        self.decay = decay
        self.enable_adaptive_interval_hook = enable_adaptive_interval_hook
        self.enable_eval_before_run = enable_eval_before_run

        self._initialized = False
        self._original_interval = None

    def before_run(self, runner):
        """Before run."""
        if self.enable_eval_before_run:
            hook = self.get_evalhook(runner)
            if hook is None:
                logger.warning("EvalHook is not found in runner. Skipping enabling evaluation before run.")
                return
            self._original_interval = hook.interval
            hook.interval = 1
            hook.start = 0

    def before_train_iter(self, runner):
        """Before train iter."""
        if self.enable_eval_before_run and self._original_interval is not None:
            hook = self.get_evalhook(runner)
            hook.interval = self._original_interval
            self._original_interval = None

        if self.enable_adaptive_interval_hook and not self._initialized:
            self.max_interval = min(self.max_interval, runner.max_epochs - runner.epoch)
            iter_per_epoch = len(runner.data_loader)
            adaptive_interval = self.get_adaptive_interval(iter_per_epoch)
            for hook in runner.hooks:
                if isinstance(hook, EvalHook):
                    # make sure evaluation is done at last to save best checkpoint
                    limit = runner.max_epochs if hook.by_epoch else runner.max_iters
                    adaptive_interval = min(adaptive_interval, limit)
                    logger.info(f"Update EvalHook interval: {hook.interval} -> {adaptive_interval}")
                    hook.interval = adaptive_interval
                elif isinstance(hook, LrUpdaterHook):
                    if hasattr(hook, "interval") and hasattr(hook, "patience"):
                        hook.interval = adaptive_interval
                        logger.info(f"Update LrUpdaterHook interval: {hook.interval} -> {adaptive_interval}")
                elif isinstance(hook, EarlyStoppingHook):
                    logger.info(f"Update EarlyStoppingHook interval: {hook.interval} -> {adaptive_interval}")
                    hook.start = adaptive_interval
                    hook.interval = adaptive_interval
                elif isinstance(hook, CheckpointHook):
                    # make sure checkpoint is saved at last
                    limit = runner.max_epochs if hook.by_epoch else runner.max_iters
                    adaptive_interval = min(adaptive_interval, limit)
                    logger.info(f"Update CheckpointHook interval: {hook.interval} -> {adaptive_interval}")
                    hook.interval = adaptive_interval
            self._initialized = True

    def get_adaptive_interval(self, iter_per_epoch):
        """Get adaptive interval."""
        adaptive_interval = max(round(math.exp(self.decay * iter_per_epoch) * self.max_interval), 1)
        return adaptive_interval

    def get_evalhook(self, runner):
        """Get evaluation hook."""
        target_hook = None
        for hook in runner.hooks:
            if isinstance(hook, EvalHook):
                assert target_hook is None, "More than 1 EvalHook is found in runner."
                target_hook = hook
        return target_hook
