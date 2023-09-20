"""Cancel hooks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import os
from typing import Callable

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import EpochBasedTrainLoop, Runner

from otx.v2.api.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-instance-attributes, protected-access, too-many-arguments, unused-argument
@HOOKS.register_module()
class CancelTrainingHook(Hook):
    """CancelTrainingHook for Training Stopping."""

    def __init__(self, interval: int = 5) -> None:
        """Periodically check whether whether a stop signal is sent to the runner during model training.

        Every 'check_interval' iterations, the work_dir for the runner is checked to see if a file '.stop_training'
        is present. If it is, training is stopped.

        :param interval: Period for checking for stop signal, given in iterations.

        """
        self.interval = interval

    @staticmethod
    def _check_for_stop_signal(runner: Runner):
        """Log _check_for_stop_signal for CancelTrainingHook."""
        work_dir = runner.work_dir
        stop_filepath = os.path.join(work_dir, ".stop_training")
        if os.path.exists(stop_filepath):
            if isinstance(runner._train_loop, EpochBasedTrainLoop):
                epoch = runner._train_loop.epoch
                runner._max_epochs = epoch  # Force runner to stop by pretending it has reached it's max_epoch
            runner.should_stop = True  # Set this flag to true to stop the current training epoch
            os.remove(stop_filepath)

    def after_train_iter(self, runner: Runner):
        """Log after_train_iter for CancelTrainingHook."""
        if not self.every_n_iters(runner, self.interval):
            return
        self._check_for_stop_signal(runner)


@HOOKS.register_module()
class CancelInterfaceHook(Hook):
    """Cancel interface. If called, running job will be terminated."""

    def __init__(self, init_callback: Callable, interval=5) -> None:
        self.on_init_callback = init_callback
        self.runner = None
        self.interval = interval

    def cancel(self):
        """Cancel."""
        logger.info("CancelInterfaceHook.cancel() is called.")
        if self.runner is None:
            logger.warning("runner is not configured yet. ignored this request.")
            return

        if self.runner.should_stop:
            logger.warning("cancel already requested.")
            return

        if isinstance(self.runner._train_loop, EpochBasedTrainLoop):
            epoch = self.runner._train_loop.epoch
            self.runner._max_epochs = epoch  # Force runner to stop by pretending it has reached it's max_epoch
        self.runner.should_stop = True  # Set this flag to true to stop the current training epoch
        logger.info("requested stopping to the runner")

    def before_run(self, runner):
        """Before run."""
        self.runner = runner
        self.on_init_callback(self)

    def after_run(self, runner):
        """After run."""
        self.runner = None
