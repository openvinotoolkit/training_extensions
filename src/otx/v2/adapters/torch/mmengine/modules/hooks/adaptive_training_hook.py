"""Adaptive training schedule hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
from typing import Optional, Sequence, Dict

from mmengine.hooks import CheckpointHook, Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner, EpochBasedTrainLoop
from mmengine.hooks import ParamSchedulerHook

from otx.v2.adapters.torch.mmengine.modules.hooks.early_stopping_hook import (
    EarlyStoppingHook,
)
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class AdaptiveTrainSchedulingHook(Hook):
    """Adaptive Training Scheduling Hook.

    Depending on the size of iteration per epoch, adaptively update the validation interval and related values.

    Args:
        base_lr_patience (int): The value of LR drop patience are expected in total epoch.
            Patience used when interval is 1, Defaults to 5.
        min_lr_patience (int): Minumum value of LR drop patience.
            Defaults to 2.
        base_es_patience (int): The value of Early-Stopping patience are expected in total epoch.
            Patience used when interval is 1, Defaults to 10.
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
        base_lr_patience=5,
        min_lr_patience=2,
        base_es_patience=10,
        min_es_patience=3,
        decay=-0.025,
        enable_adaptive_interval_hook=False,
        enable_eval_before_run=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_interval = max_interval
        self.base_lr_patience = base_lr_patience
        self.min_lr_patience = min_lr_patience
        self.base_es_patience = base_es_patience
        self.min_es_patience = min_es_patience
        self.decay = decay
        self.enable_adaptive_interval_hook = enable_adaptive_interval_hook
        self.enable_eval_before_run = enable_eval_before_run

        self._initialized = False
        self._original_interval = None

    def before_train_iter(self, runner: Runner, batch_idx: int, data_batch: Sequence[Dict]):
        """Before train iter."""
        if self.enable_eval_before_run and not self._initialized:
            runner.val_loop.run()

        if self.enable_adaptive_interval_hook and not self._initialized:
            self.max_interval = min(self.max_interval, runner.max_epochs - runner.epoch)
            iter_per_epoch = len(runner.train_dataloader)
            adaptive_interval = self.get_adaptive_interval(iter_per_epoch)
            self.update_validation_interval(runner, adaptive_interval)
            for hook in runner.hooks:
                if isinstance(hook, ParamSchedulerHook):
                    if hasattr(hook, "interval") and hasattr(hook, "patience"):
                        hook.interval = adaptive_interval
                        logger.info(f"Update ParamSchedulerHook interval: {hook.interval} -> {adaptive_interval}")
                elif isinstance(hook, EarlyStoppingHook):
                    patience = max(
                        math.ceil((self.base_es_patience / adaptive_interval)),
                        self.min_es_patience,
                    )
                    logger.info(f"Update EarlyStoppingHook patience: {hook.patience} -> {patience}")
                    hook.start = adaptive_interval
                    hook.interval = adaptive_interval
                    hook.patience = patience
                elif isinstance(hook, CheckpointHook):
                    # make sure checkpoint is saved at last
                    limit = runner.max_epochs if hook.by_epoch else runner.max_iters
                    adaptive_interval = min(adaptive_interval, limit)
                    logger.info(f"Update CheckpointHook interval: {hook.interval} -> {adaptive_interval}")
                    hook.interval = adaptive_interval
            self._initialized = True

    def get_adaptive_interval(self, iter_per_epoch: int) -> int:
        """Get adaptive interval."""
        adaptive_interval = max(round(math.exp(self.decay * iter_per_epoch) * self.max_interval), 1)
        return adaptive_interval

    def update_validation_interval(self, runner: Runner, adaptive_interval: int):
        """Update validation interval of training loop."""
        # make sure evaluation is done at last to save best checkpoint
        limit = runner.max_epochs if isinstance(runner.train_loop, EpochBasedTrainLoop) else runner.max_iters
        adaptive_interval = min(adaptive_interval, limit)
        logger.info(f"Update Validation interval: {runner.train_loop.val_interval} -> {adaptive_interval}")
        runner.train_loop.val_interval = adaptive_interval
        runner.train_loop.dynamic_intervals = [adaptive_interval]
