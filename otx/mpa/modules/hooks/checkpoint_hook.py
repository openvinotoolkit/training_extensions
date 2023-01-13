# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Copyright (c) Open-MMLab. All rights reserved.
import os

from mmcv.runner.dist_utils import allreduce_params, master_only
from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class CheckpointHookWithValResults(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        sync_buffer (bool): Whether to synchronize buffers in different
            gpus. Default: False.
    """

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 sync_buffer=False,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs
        self.sync_buffer = sync_buffer

    def before_run(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return

        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        if getattr(runner, "save_ckpt", False):
            runner.logger.info(f'Saving best checkpoint at {runner.epoch + 1} epochs')
            self._save_checkpoint(runner)
            runner.save_ckpt = False
        self._save_last_epoch(runner)

    @master_only
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        cur_ckpt_filename = f'best_epoch_{runner.epoch + 1}.pth'
        runner.save_checkpoint(
            self.out_dir, filename_tmpl=cur_ckpt_filename, save_optimizer=self.save_optimizer, **self.args)
        if runner.meta is not None:
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['best_ckpt'] = os.path.join(
                self.out_dir, cur_ckpt_filename)

    @master_only
    def _save_last_epoch(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get('filename_tmpl', f'epoch_{runner.epoch + 1}.pth')
            else:
                cur_ckpt_filename = self.args.get('filename_tmpl', f'iter_{runner.iter + 1}.pth')
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = os.path.join(
                self.out_dir, cur_ckpt_filename)
        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = os.path.join(self.out_dir,
                                         filename_tmpl.format(_step))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        if hasattr(runner, 'save_ckpt'):
            if runner.save_ckpt:
                runner.logger.info(
                    f'Saving checkpoint at {runner.iter + 1} iterations')
                if self.sync_buffer:
                    allreduce_params(runner.model.buffers())
                self._save_checkpoint(runner)
            runner.save_ckpt = False
