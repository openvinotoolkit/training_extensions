"""CheckpointHook with validation results for classification task."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Copyright (c) Open-MMLab. All rights reserved.
from pathlib import Path
from typing import Optional, Union

from mmengine.dist import all_reduce_params, master_only
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class CheckpointHookWithValResults(Hook):  # pylint: disable=too-many-instance-attributes
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

    def __init__(
        self,
        interval: int = -1,
        by_epoch: bool = True,
        save_optimizer: bool = True,
        out_dir: Optional[Union[str, Path]] = None,
        max_keep_ckpts: int = -1,
        sync_buffer: bool = False,
        **kwargs,
    ) -> None:
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self._best_model_weight: Optional[Path] = None

    def after_train_epoch(self, runner: Runner, **kwargs) -> None:
        """Checkpoint stuffs after train epoch."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return

        if self.sync_buffer:
            all_reduce_params(runner.model.buffers())
        save_ema_model = hasattr(runner, "save_ema_model") and runner.save_ema_model
        if save_ema_model:
            backup_model = runner.model
            runner.model = runner.ema_model
        if getattr(runner, "save_ckpt", False):
            runner.logger.info(f"Saving best checkpoint at {runner.epoch + 1} epochs")
            self._save_best_checkpoint(runner)
            runner.save_ckpt = False

        self._save_latest_checkpoint(runner)

        if save_ema_model:
            runner.model = backup_model
            runner.save_ema_model = False

    @master_only
    def _save_best_checkpoint(self, runner: Runner) -> None:
        """Save the current checkpoint and delete unwanted checkpoint."""
        if self.out_dir is None:
            self.out_dir = runner.work_dir
        if self._best_model_weight is not None:  # remove previous best model weight
            prev_model_weight = Path(self.out_dir) / self._best_model_weight
            if prev_model_weight.exists():
                prev_model_weight.unlink()

        weight_name = f"best_epoch_{runner.epoch + 1}.pth" if self.by_epoch else f"best_iter_{runner.iter + 1}.pth"
        runner.save_checkpoint(self.out_dir, filename=weight_name, save_optimizer=self.save_optimizer, **self.args)

        self._best_model_weight = Path(weight_name)
        if runner.meta is not None:
            runner.meta.setdefault("hook_msgs", {})
            runner.meta["hook_msgs"]["best_ckpt"] = str(self.out_dir / self._best_model_weight)

    @master_only
    def _save_latest_checkpoint(self, runner: Runner) -> None:
        """Save the current checkpoint and delete unwanted checkpoint."""
        if self.out_dir is None:
            self.out_dir = runner.work_dir
        if self.by_epoch:
            weight_name_format = "epoch_{}.pth"
            cur_step = runner.epoch + 1
        else:
            weight_name_format = "iter_{}.pth"
            cur_step = runner.iter + 1

        runner.save_checkpoint(
            self.out_dir,
            filename=weight_name_format.format(cur_step),
            save_optimizer=self.save_optimizer,
            **self.args,
        )

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            for _step in range(cur_step - self.max_keep_ckpts * self.interval, 0, -self.interval):
                ckpt_path = Path(self.out_dir) / Path(weight_name_format.format(_step))
                if ckpt_path.exists():
                    ckpt_path.unlink()

        if runner.meta is not None:
            cur_ckpt_filename = Path(self.args.get("filename_tmpl", weight_name_format.format(cur_step)))
            runner.meta.setdefault("hook_msgs", {})
            runner.meta["hook_msgs"]["last_ckpt"] = str(Path(self.out_dir) / cur_ckpt_filename)

    def after_train_iter(self, runner: Runner, **kwargs) -> None:
        """Checkpoint stuffs after train iteration."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        if hasattr(runner, "save_ckpt"):
            if runner.save_ckpt:
                runner.logger.info(f"Saving checkpoint at {runner.iter + 1} iterations")
                if self.sync_buffer:
                    all_reduce_params(runner.model.buffers())
                self._save_checkpoint(runner)
            runner.save_ckpt = False


@HOOKS.register_module()
class EnsureCorrectBestCheckpointHook(Hook):
    """EnsureCorrectBestCheckpointHook.

    This hook makes sure that the 'best_mAP' checkpoint points properly to the best model, even if the best model is
    created in the last epoch.
    """

    def after_run(self, runner: Runner) -> None:
        """Called after train epoch hooks."""
        runner.call_hook("after_train_epoch")


@HOOKS.register_module()
class SaveInitialWeightHook(Hook):
    """Save the initial weights before training."""

    def __init__(self, save_path: str, file_name: str = "weights.pth", **kwargs) -> None:
        self._save_path = save_path
        self._file_name = file_name
        self._args = kwargs

    def before_run(self, runner: Runner) -> None:
        """Save initial the weights before training."""
        runner.logger.info("Saving weight before training")
        runner.save_checkpoint(
            self._save_path,
            filename=self._file_name,
            save_optimizer=False,
            create_symlink=False,
            **self._args,
        )
