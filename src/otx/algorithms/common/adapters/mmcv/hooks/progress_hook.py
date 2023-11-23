"""Collections of hooks for common OTX algorithms."""

# Copyright (C) 2021-2023 Intel Corporation
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

import math

from mmcv.runner import BaseRunner
from mmcv.runner.hooks import HOOKS, Hook

from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from otx.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class OTXProgressHook(Hook):
    """OTXProgressHook for getting progress."""

    def __init__(self, time_monitor: TimeMonitorCallback, verbose: bool = False):
        super().__init__()
        self.time_monitor = time_monitor
        self.verbose = verbose
        self.print_threshold = 1

    def before_run(self, runner: BaseRunner):
        """Called before_run in OTXProgressHook."""
        total_epochs = runner.max_epochs if runner.max_epochs is not None else 1
        self.time_monitor.total_epochs = total_epochs
        self.time_monitor.train_steps = runner.max_iters // total_epochs if total_epochs else 1
        self.time_monitor.steps_per_epoch = self.time_monitor.train_steps + self.time_monitor.val_steps
        self.time_monitor.total_steps = max(math.ceil(self.time_monitor.steps_per_epoch * total_epochs), 1)
        self.time_monitor.current_step = 0
        self.time_monitor.current_epoch = 0
        self.time_monitor.on_train_begin()

    def before_epoch(self, runner: BaseRunner):
        """Called before_epoch in OTXProgressHook."""
        self.time_monitor.on_epoch_begin(runner.epoch)

    def after_epoch(self, runner: BaseRunner):
        """Called after_epoch in OTXProgressHook."""
        # put some runner's training status to use on the other hooks
        runner.log_buffer.output["current_iters"] = runner.iter
        self.time_monitor.on_epoch_end(runner.epoch, runner.log_buffer.output)

    def before_iter(self, runner: BaseRunner):
        """Called before_iter in OTXProgressHook."""
        self.time_monitor.on_train_batch_begin(1)

    def after_iter(self, runner: BaseRunner):
        """Called after_iter in OTXProgressHook."""
        # put some runner's training status to use on the other hooks
        runner.log_buffer.output["current_iters"] = runner.iter
        self.time_monitor.on_train_batch_end(1)
        if self.verbose:
            progress = self.progress
            if progress >= self.print_threshold:
                logger.info(f"training progress {progress:.0f}%")
                self.print_threshold = (progress + 10) // 10 * 10

    def before_val_iter(self, runner: BaseRunner):
        """Called before_val_iter in OTXProgressHook."""
        self.time_monitor.on_test_batch_begin(1, logger)

    def after_val_iter(self, runner: BaseRunner):
        """Called after_val_iter in OTXProgressHook."""
        self.time_monitor.on_test_batch_end(1, logger)

    def after_run(self, runner: BaseRunner):
        """Called after_run in OTXProgressHook."""
        self.time_monitor.on_train_end(1)
        if self.time_monitor.update_progress_callback:
            self.time_monitor.update_progress_callback(int(self.time_monitor.get_progress()))

    @property
    def progress(self):
        """Getting Progress from time monitor."""
        return self.time_monitor.get_progress()
