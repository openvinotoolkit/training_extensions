"""EMA hooks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
from math import cos, pi

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, BaseRunner, Hook
from mmcv.runner.hooks.ema import EMAHook

from otx.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class CustomModelEMAHook(EMAHook):
    """Custom EMAHook to update momentum for ema over training."""

    def __init__(self, momentum=0.0002, epoch_momentum=0.0, interval=1, **kwargs):
        super().__init__(momentum=momentum, interval=interval, **kwargs)
        self.momentum = momentum
        self.epoch_momentum = epoch_momentum
        self.interval = interval

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model
        """
        if is_module_wrapper(runner.model):
            model = runner.model.module.model_s if hasattr(runner.model.module, "model_s") else runner.model.module
        else:
            model = runner.model.model_s if hasattr(runner.model, "model_s") else runner.model
        self.param_ema_buffer = {}
        self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    def before_train_epoch(self, runner):
        """Update the momentum."""
        if self.epoch_momentum > 0.0:
            iter_per_epoch = len(runner.data_loader)
            epoch_decay = 1 - self.epoch_momentum
            iter_decay = math.pow(epoch_decay, self.interval / iter_per_epoch)
            self.momentum = 1 - iter_decay
            logger.info(f"Update EMA momentum: {self.momentum}")
            self.epoch_momentum = 0.0  # disable re-compute

        super().before_train_epoch(runner)


@HOOKS.register_module()
class EMAMomentumUpdateHook(Hook):
    """Exponential moving average (EMA) momentum update hook for self-supervised methods.

    This hook includes momentum adjustment in self-supervised methods following:
        m = 1 - ( 1- m_0) * (cos(pi * k / K) + 1) / 2,
        k: current step, K: total steps.

    :param end_momentum: The final momentum coefficient for the target network, defaults to 1.
    :param update_interval: Interval to update new momentum, defaults to 1.
    :param by_epoch: Whether updating momentum by epoch or not, defaults to False.
    """

    def __init__(self, end_momentum: float = 1.0, update_interval: int = 1, by_epoch: bool = False):
        self.by_epoch = by_epoch
        self.end_momentum = end_momentum
        self.update_interval = update_interval

    def before_train_epoch(self, runner: BaseRunner):
        """Called before_train_epoch in EMAMomentumUpdateHook."""
        if not self.by_epoch:
            return

        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        if not hasattr(model, "momentum"):
            raise AttributeError('The model must have attribute "momentum".')
        if not hasattr(model, "base_momentum"):
            raise AttributeError('The model must have attribute "base_momentum".')

        if self.every_n_epochs(runner, self.update_interval):
            cur_epoch = runner.epoch
            max_epoch = runner.max_epochs
            base_m = model.base_momentum
            updated_m = (
                self.end_momentum - (self.end_momentum - base_m) * (cos(pi * cur_epoch / float(max_epoch)) + 1) / 2
            )
            model.momentum = updated_m

    def before_train_iter(self, runner: BaseRunner):
        """Called before_train_iter in EMAMomentumUpdateHook."""
        if self.by_epoch:
            return

        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        if not hasattr(model, "momentum"):
            raise AttributeError('The model must have attribute "momentum".')
        if not hasattr(model, "base_momentum"):
            raise AttributeError('The model must have attribute "base_momentum".')

        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            max_iter = runner.max_iters
            base_m = model.base_momentum
            updated_m = (
                self.end_momentum - (self.end_momentum - base_m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2
            )
            model.momentum = updated_m

    def after_train_iter(self, runner: BaseRunner):
        """Called after_train_iter in EMAMomentumUpdateHook."""
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                runner.model.module.momentum_update()
            else:
                runner.model.momentum_update()
