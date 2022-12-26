# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import torch
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class CompressionHook(Hook):
    def __init__(self, compression_ctrl=None):
        self.compression_ctrl = compression_ctrl

    def after_train_iter(self, runner):
        self.compression_ctrl.scheduler.step()

    def after_train_epoch(self, runner):
        self.compression_ctrl.scheduler.epoch_step()
        if runner.rank == 0:
            runner.logger.info(self.compression_ctrl.statistics().to_str())

    def before_run(self, runner):
        runner.compression_ctrl = self.compression_ctrl
        if runner.rank == 0:
            runner.logger.info(self.compression_ctrl.statistics().to_str())

    @master_only
    def after_run(self, runner):
        compression_state = self.compression_ctrl.get_compression_state()
        for algo_state in compression_state.get("ctrl_state", {}).values():
            if not algo_state.get("scheduler_state"):
                algo_state["scheduler_state"] = {"current_step": 0, "current_epoch": 0}
        torch.save(
            compression_state,
            os.path.join(runner.work_dir, "compression_state.pth")
        )


@HOOKS.register_module()
class CheckpointHookBeforeTraining(Hook):
    """Save checkpoints before training.

    Args:
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
    """

    def __init__(self, out_dir=None, **kwargs):
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def before_run(self, runner):
        runner.logger.info(f"Saving checkpoint before training")
        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir,
            filename_tmpl="before_training.pth",
            **self.args,
        )
