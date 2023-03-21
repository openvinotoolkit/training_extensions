"""Collections of hooks for common OTX algorithms."""

# Copyright (C) 2021-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from math import cos, pi

from mmcv.parallel import is_module_wrapper
from mmcv.runner import BaseRunner
from mmcv.runner.hooks import HOOKS, Hook

from otx.mpa.utils.logger import get_logger

logger = get_logger()


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

    def __init__(self, end_momentum: float = 1.0, update_interval: int = 1, by_epoch: bool = False, **kwargs):
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
