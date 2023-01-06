# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math

from mmcv.runner import HOOKS, Hook, LrUpdaterHook
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.evaluation import EvalHook

from otx.mpa.modules.hooks.early_stopping_hook import EarlyStoppingHook
from otx.mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class AdaptiveTrainSchedulingHook(Hook):
    """Adaptive Training Scheduling Hook

    Depending on the size of iteration per epoch, adaptively update the validation interval and related values.

    Args:
        max_interval (int): Maximum value of validation interval.
            Defaults to 5.
        base_lr_patience (int): The value of LR drop patience are expected in total epoch.
            Patience used when interval is 1, Defaults to 5.
        min_lr_patience (int): Minumum value of LR drop patience.
            Defaults to 2.
        base_es_patience (int): The value of Early-Stopping patience are expected in total epoch.
            Patience used when interval is 1, Defaults to 10.
    """

    def __init__(
        self,
        max_interval=5,
        base_lr_patience=5,
        min_lr_patience=2,
        base_es_patience=10,
        min_es_patience=3,
        decay=-0.025,
        eval_before_train=False,
        eval_after_train=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_interval = max_interval
        self.base_lr_patience = base_lr_patience
        self.min_lr_patience = min_lr_patience
        self.base_es_patience = base_es_patience
        self.min_es_patience = min_es_patience
        self.decay = decay
        self.eval_before_train = eval_before_train
        self.eval_after_train = eval_after_train
        self.enabled = False
        self.initialized = False

        self._done_eval_before_train = False

    def before_train_epoch(self, runner):
        if self.eval_before_train and not self._done_eval_before_train:
            for hook in runner.hooks:
                if isinstance(hook, EvalHook):
                    hook.start = 0
                    hook.interval = 1
            self._done_eval_before_train = True

    def before_train_iter(self, runner):
        if not self.initialized:
            iter_per_epoch = len(runner.data_loader)
            adaptive_interval = self.get_adaptive_interval(iter_per_epoch)
            for hook in runner.hooks:
                if isinstance(hook, EvalHook):
                    logger.info(f"Update EvalHook interval: {hook.interval} -> {adaptive_interval}")
                    hook.interval = adaptive_interval
                elif isinstance(hook, LrUpdaterHook):
                    patience = max(
                        math.ceil((self.base_lr_patience / adaptive_interval)),
                        self.min_lr_patience,
                    )
                    logger.info(f"Update LrUpdaterHook patience: {hook.patience} -> {patience}")
                    hook.interval = adaptive_interval
                    hook.patience = patience
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
                    logger.info(f"Update CheckpointHook interval: {hook.interval} -> {adaptive_interval}")
                    hook.interval = adaptive_interval
            self.initialized = True

    def after_train_epoch(self, runner):
        if runner.iter >= runner.max_iters:
            for hook in runner.hooks:
                if isinstance(hook, EvalHook):
                    hook.start = 0
                    hook.interval = 1

    def get_adaptive_interval(self, iter_per_epoch):
        adaptive_interval = max(round(math.exp(self.decay * iter_per_epoch) * self.max_interval), 1)
        return adaptive_interval
