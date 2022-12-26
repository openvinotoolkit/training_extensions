# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
from mmcv.runner import HOOKS, Hook, LrUpdaterHook
from mmcv.runner.hooks.checkpoint import CheckpointHook
from otx.mpa.utils.logger import get_logger
from otx.mpa.modules.hooks.early_stopping_hook import EarlyStoppingHook

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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_interval = max_interval
        self.base_lr_patience = base_lr_patience
        self.min_lr_patience = min_lr_patience
        self.base_es_patience = base_es_patience
        self.min_es_patience = min_es_patience
        self.decay = decay
        self.initialized = False
        self.enabled = False

    def before_train_epoch(self, runner):
        if not self.initialized:
            iter_per_epoch = len(runner.data_loader)
            adaptive_interval = self.get_adaptive_interval(iter_per_epoch)
            for hook in runner.hooks:
                if 'EvalHook' in str(hook):
                    hook.interval = adaptive_interval
                    logger.info(f"Update Validation Interval: {adaptive_interval}")
                elif isinstance(hook, LrUpdaterHook):
                    hook.interval = adaptive_interval
                    hook.patience = max(
                        math.ceil((self.base_lr_patience / adaptive_interval)),
                        self.min_lr_patience,
                    )
                    logger.info(f"Update Lr patience: {hook.patience}")
                elif isinstance(hook, EarlyStoppingHook):
                    hook.start = adaptive_interval
                    hook.interval = adaptive_interval
                    hook.patience = max(
                        math.ceil((self.base_es_patience / adaptive_interval)),
                        self.min_es_patience,
                    )
                    logger.info(f"Update Early-Stop patience: {hook.patience}")
                elif isinstance(hook, CheckpointHook):
                    hook.interval = adaptive_interval
            self.initialized = True

    def get_adaptive_interval(self, iter_per_epoch):
        adaptive_interval = max(
            round(math.exp(self.decay * iter_per_epoch) * self.max_interval), 1
        )
        return adaptive_interval
