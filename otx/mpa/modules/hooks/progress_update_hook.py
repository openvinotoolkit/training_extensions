# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook

from otx.mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class ProgressUpdateHook(Hook):
    def __init__(self, name, callback, **kwargs):
        logger.info("init() ProgressUpdateHook")
        self.update_callback = callback
        self.name = name
        self._progress = 0.0

    def after_epoch(self, runner):
        self._progress = float(runner.epoch) / runner.max_epochs
        self.update_progress(runner)

    def after_iter(self, runner):
        self._progress = float(runner.iter) / runner.max_iters
        self.update_progress(runner)

    def update_progress(self, runner):
        if callable(self.update_callback):
            self.update_callback(progress=int(self.progress * 100))
        logger.debug(f"[{self.name}] update progress {int(self.progress * 100)}")

    @property
    def progress(self):
        return self._progress
