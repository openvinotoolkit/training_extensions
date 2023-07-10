"""NNCF task related hooks."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class CompressionHook(Hook):
    """CompressionHook."""

    COMPRESSION_STATE_FILE_NAME = "meta_state.pth"

    def __init__(self, compression_ctrl=None):
        self.compression_ctrl = compression_ctrl

    def after_train_iter(self, runner):
        """Called after train iter."""
        self.compression_ctrl.scheduler.step()

    def after_train_epoch(self, runner):
        """Called after train epoch."""
        self.compression_ctrl.scheduler.epoch_step()
        if runner.rank == 0:
            runner.logger.info(self.compression_ctrl.statistics().to_str())

    def before_run(self, runner):
        """Called before run."""
        runner.compression_ctrl = self.compression_ctrl
        if runner.rank == 0:
            runner.logger.info(self.compression_ctrl.statistics().to_str())
