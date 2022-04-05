# Copyright (C) 2021 Intel Corporation
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

import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict, Optional

from mmcv.runner.hooks import HOOKS, Hook, LoggerHook
from mmcv.runner import BaseRunner, EpochBasedRunner
from mmcv.runner.dist_utils import master_only

from ote_sdk.utils.argument_checks import check_input_parameters_type
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback


logger = logging.getLogger(__name__)


@HOOKS.register_module()
class CancelTrainingHook(Hook):
    @check_input_parameters_type()
    def __init__(self, interval: int = 5):
        """
        Periodically check whether whether a stop signal is sent to the runner during model training.
        Every 'check_interval' iterations, the work_dir for the runner is checked to see if a file '.stop_training'
        is present. If it is, training is stopped.

        :param interval: Period for checking for stop signal, given in iterations.

        """
        self.interval = interval

    @staticmethod
    def _check_for_stop_signal(runner: BaseRunner):
        work_dir = runner.work_dir
        stop_filepath = os.path.join(work_dir, '.stop_training')
        if os.path.exists(stop_filepath):
            if isinstance(runner, EpochBasedRunner):
                epoch = runner.epoch
                runner._max_epochs = epoch  # Force runner to stop by pretending it has reached it's max_epoch
            runner.should_stop = True  # Set this flag to true to stop the current training epoch
            os.remove(stop_filepath)

    @check_input_parameters_type()
    def after_train_iter(self, runner: BaseRunner):
        if not self.every_n_iters(runner, self.interval):
            return
        self._check_for_stop_signal(runner)


@HOOKS.register_module()
class FixedMomentumUpdaterHook(Hook):
    def __init__(self):
        """
        This hook does nothing, as the momentum is fixed by default. The hook is here to streamline switching between
        different LR schedules.
        """
        pass

    @check_input_parameters_type()
    def before_run(self, runner: BaseRunner):
        pass


@HOOKS.register_module()
class EnsureCorrectBestCheckpointHook(Hook):
    def __init__(self):
        """
        This hook makes sure that the 'best_mAP' checkpoint points properly to the best model, even if the best model is
        created in the last epoch.
        """
        pass

    @check_input_parameters_type()
    def after_run(self, runner: BaseRunner):
        runner.call_hook('after_train_epoch')


@HOOKS.register_module()
class OTELoggerHook(LoggerHook):

    class Curve:
        def __init__(self):
            self.x = []
            self.y = []

        def __repr__(self):
            points = []
            for x, y in zip(self.x, self.y):
                points.append(f'({x},{y})')
            return 'curve[' + ','.join(points) + ']'

    @check_input_parameters_type()
    def __init__(self,
                 curves: Optional[Dict[Any, Curve]] = None,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = True,
                 by_epoch: bool = True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.curves = curves if curves is not None else defaultdict(self.Curve)

    @master_only
    @check_input_parameters_type()
    def log(self, runner: BaseRunner):
        tags = self.get_loggable_tags(runner, allow_text=False)
        if runner.max_epochs is not None:
            normalized_iter = self.get_iter(runner) / runner.max_iters * runner.max_epochs
        else:
            normalized_iter = self.get_iter(runner)
        for tag, value in tags.items():
            curve = self.curves[tag]
            # Remove duplicates.
            if len(curve.x) > 0 and curve.x[-1] == normalized_iter:
                curve.x.pop()
                curve.y.pop()
            curve.x.append(normalized_iter)
            curve.y.append(value)

    @check_input_parameters_type()
    def after_train_epoch(self, runner: BaseRunner):
        # Iteration counter is increased right after the last iteration in the epoch,
        # temporarily decrease it back.
        runner._iter -= 1
        super().after_train_epoch(runner)
        runner._iter += 1


@HOOKS.register_module()
class OTEProgressHook(Hook):
    @check_input_parameters_type()
    def __init__(self, time_monitor: TimeMonitorCallback, verbose: bool = False):
        super().__init__()
        self.time_monitor = time_monitor
        self.verbose = verbose
        self.print_threshold = 1

    @check_input_parameters_type()
    def before_run(self, runner: BaseRunner):
        total_epochs = runner.max_epochs if runner.max_epochs is not None else 1
        self.time_monitor.total_epochs = total_epochs
        self.time_monitor.train_steps = runner.max_iters // total_epochs if total_epochs else 1
        self.time_monitor.steps_per_epoch = self.time_monitor.train_steps + self.time_monitor.val_steps
        self.time_monitor.total_steps = max(math.ceil(self.time_monitor.steps_per_epoch * total_epochs), 1)
        self.time_monitor.current_step = 0
        self.time_monitor.current_epoch = 0
        self.time_monitor.on_train_begin()

    @check_input_parameters_type()
    def before_epoch(self, runner: BaseRunner):
        self.time_monitor.on_epoch_begin(runner.epoch)

    @check_input_parameters_type()
    def after_epoch(self, runner: BaseRunner):
        self.time_monitor.on_epoch_end(runner.epoch, runner.log_buffer.output)

    @check_input_parameters_type()
    def before_iter(self, runner: BaseRunner):
        self.time_monitor.on_train_batch_begin(1)

    @check_input_parameters_type()
    def after_iter(self, runner: BaseRunner):
        self.time_monitor.on_train_batch_end(1)
        if self.verbose:
            progress = self.progress
            if progress >= self.print_threshold:
                logger.warning(f'training progress {progress:.0f}%')
                self.print_threshold = (progress + 10) // 10 * 10

    @check_input_parameters_type()
    def before_val_iter(self, runner: BaseRunner):
        self.time_monitor.on_test_batch_begin(1)

    @check_input_parameters_type()
    def after_val_iter(self, runner: BaseRunner):
        self.time_monitor.on_test_batch_end(1)

    @check_input_parameters_type()
    def after_run(self, runner: BaseRunner):
        self.time_monitor.on_train_end(1)
        self.time_monitor.update_progress_callback(self.time_monitor.get_progress())

    @property
    def progress(self):
        return self.time_monitor.get_progress()
