"""Train function for segmentation task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
import logging
from math import inf
from typing import Union, Optional, List
from pathlib import Path

import torch
import tqdm
from mmcv.runner.checkpoint import save_checkpoint as mmcv_save_checkpoint
from mmseg.apis import single_gpu_test
from mmseg.core import build_optimizer
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
from mmseg.utils.util_distribution import build_dp, dp_factory

from otx.algorithms.common.adapters.mmcv.utils import XPUDataParallel

dp_factory["xpu"] = XPUDataParallel
logger = get_root_logger(logging.INFO)


def train_segmentor(model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    """Launch segmentor training."""
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True,
    )
    # The overall dataloader settings
    loader_cfg.update(
        {
            k: v
            for k, v in cfg.data.items()
            if k not in ["train", "val", "test", "train_dataloader", "val_dataloader", "test_dataloader"]
        }
    )

    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get("train_dataloader", {})}
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on devices
    if cfg.device == "xpu":
        use_autocast = bool(cfg.get("fp16_", False))
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids, enable_autocast=use_autocast)
        model.to(f"xpu:{cfg.gpu_ids[0]}")
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.device == "xpu":
        fp16_cfg = cfg.get("fp16_", None)
        if fp16_cfg is not None:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        model.train()
        model, optimizer = torch.xpu.optimize(model, optimizer=optimizer, dtype=dtype)

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            "samples_per_gpu": 1,
            "shuffle": False,  # Not shuffle by default
            **cfg.data.get("val_dataloader", {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)

    # make lr updater hook
    del cfg.lr_config["policy"]
    lr_hook = ReduceLROnPlateauLrUpdaterHook(optimizer=optimizer, **cfg.lr_config)

    iter_per_epoch = len(data_loaders[0])

    # Simple training loop
    iter_idx = 0
    for epoch in range(cfg.runner.max_epochs):
        # train step
        logger.info(f"Epoch #{epoch+1} training starts")
        lr_hook.register_progress(epoch, iter_idx)
        model.train()
        lr_hook.before_train_epoch()
        for i, data_batch in enumerate(data_loaders[0]):
            lr_hook.register_progress(epoch, iter_idx)
            lr_hook.before_train_iter()
            train_outputs = model.train_step(data_batch, optimizer)
            iter_idx += 1

            # backward & optimization
            optimizer.zero_grad()
            train_outputs['loss'].backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                logger.info(
                    f"[{i+1:^3}/ {iter_per_epoch:^3} ] " + \
                    " / ".join([f"{key} : {val}" for key, val in train_outputs['log_vars'].items()])
                )

        # eval step
        model.eval()
        logger.info(f"Epoch #{epoch + 1} evaluation starts")
        if validate:
            eval_result = single_gpu_test(model, val_dataloader, show=False)
            eval_res = val_dataloader.dataset.evaluate(eval_result, logger=logger, metric='mDice', show_log=True)
            lr_hook.register_score(eval_res["mDice"])

        save_checkpoint(model, optimizer, cfg.work_dir, epoch+1)


def save_checkpoint(model, optim, out_dir: Union[str, Path], epoch: int, max_keep_ckpts=1):
    """Save the current checkpoint and delete unwanted checkpoint."""
    out_dir: Path = Path(out_dir)
    name = 'epoch_{}.pth'

    filename = name.format(epoch)
    filepath = out_dir / filename
    mmcv_save_checkpoint(model, str(filepath), optimizer=optim)
    # in some environments, `os.symlink` is not supported, you may need to
    # set `create_symlink` to False
    dst_file = out_dir / 'latest.pth'
    if dst_file.exists():
        dst_file.unlink()
    dst_file.symlink_to(filename)

    # remove other checkpoints
    if max_keep_ckpts > 0:
        redundant_ckpts = range(epoch - max_keep_ckpts, 0, -1)
        for _step in redundant_ckpts:
            ckpt_path = out_dir / name.format(_step)
            if ckpt_path.exists():
                ckpt_path.unlink()
            else:
                break


class ReduceLROnPlateauLrUpdaterHook:
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
    This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’
    number of epochs, the learning rate is reduced.

    :param min_lr: minimum learning rate. The lower bound of the desired learning rate.
    :param interval: the number of intervals for checking the hook. The interval number should be
                     the same as the evaluation interval - the `interval` variable set in
                     `evaluation` config.
    :param metric: the metric name to be monitored
    :param rule: greater or less.  In `less` mode, learning rate will be dropped if the metric has
                 stopped decreasing and in `greater` mode it will be dropped when the metric has
                 stopped increasing.
    :param patience: Number of epochs with no improvement after which learning rate will be reduced.
                     For example, if patience = 2, then we will ignore the first 2 epochs with no
                     improvement, and will only drop LR after the 3rd epoch if the metric still
                     hasn’t improved then
    :param iteration_patience: Number of iterations must be trained after the last improvement
                               before LR drops. The same as patience but the LR remains the same if
                               the number of iteration is lower than iteration_patience. This
                               variable makes sure a model is trained enough for some iterations
                               after the last improvement before dropping the LR.
    :param factor: Factor to be multiply with the learning rate.
                   For example, new_lr = current_lr * factor
    """

    rule_map = {"greater": lambda x, y: x >= y, "less": lambda x, y: x < y}
    init_value_map = {"greater": -inf, "less": inf}
    greater_keys = ["acc", "top", "AR@", "auc", "precision", "mAP", "mDice", "mIoU", "mAcc", "aAcc", "MHAcc"]
    less_keys = ["loss"]

    def __init__(
        self,
        optimizer,
        min_lr: float,
        interval: int,
        metric: str = "bbox_mAP",
        rule: Optional[str] = None,
        factor: float = 0.1,
        patience: int = 3,
        iteration_patience: int = 300,
        by_epoch: bool = True,
        warmup: Optional[str] = None,
        warmup_iters: int = 0,
        warmup_ratio: float = 0.1,
        warmup_by_epoch: bool = False) -> None:
        self.optimizer = optimizer
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant", "linear" and "exp"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters: Optional[int] = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs: Optional[int] = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr: Union[list, dict] = []  # initial lr for all param groups
        self.regular_lr: list = []  # expected lr if no warming up is performed
        self.interval = interval
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.iteration_patience = iteration_patience
        self.metric = metric
        self.bad_count = 0
        self.bad_count_iter = 0
        self.last_iter = 0
        self.current_lr = -1.0
        self.base_lr: List[float] = []
        self._init_rule(rule, metric)
        self.best_score = self.init_value_map[self.rule]
        self.current_iter = None
        self.current_epoch = None
        self.current_score = None

        self.before_run()

    def register_progress(self, epoch, iter):
        self.current_epoch = epoch
        self.current_iter = iter

    def register_score(self, score):
        self.current_score = score

    def _init_rule(self, rule, key_indicator):
        """Initialize rule, key_indicator, comparison_func, and best score.

        Here is the rule to determine which rule is used for key indicator
        when the rule is not specific:
        1. If the key indicator is in ``self.greater_keys``, the rule will be
           specified as 'greater'.
        2. Or if the key indicator is in ``self.less_keys``, the rule will be
           specified as 'less'.
        3. Or if the key indicator is equal to the substring in any one item
           in ``self.greater_keys``, the rule will be specified as 'greater'.
        4. Or if the key indicator is equal to the substring in any one item
           in ``self.less_keys``, the rule will be specified as 'less'.

        Args:
            rule (str | None): Comparison rule for best score.
            key_indicator (str | None): Key indicator to determine the
                comparison rule.
        """
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f"rule must be greater, less or None, " f"but got {rule}.")

        if rule is None:
            if key_indicator in self.greater_keys or any(key in key_indicator for key in self.greater_keys):
                rule = "greater"
            elif key_indicator in self.less_keys or any(key in key_indicator for key in self.less_keys):
                rule = "less"
            else:
                raise ValueError(
                    f"Cannot infer the rule for key " f"{key_indicator}, thus a specific rule " f"must be specified."
                )
        self.rule = rule
        self.key_indicator = key_indicator
        self.compare_func = self.rule_map[self.rule]

    def _is_check_timing(self) -> bool:
        """Check whether current epoch or iter is multiple of self.interval, skip during warmup interations."""
        return self.after_each_n_epochs(self.interval) and (self.warmup_iters <= self.current_iter)

    def after_each_n_epochs(self, interval: int) -> bool:
        """Check whether current epoch is a next epoch after multiples of interval."""
        return self.current_epoch % interval == 0 if interval > 0 and self.current_epoch != 0 else False

    def get_lr(self, base_lr: float):
        """Called get_lr in ReduceLROnPlateauLrUpdaterHook."""
        if self.current_lr < 0:
            self.current_lr = base_lr

        # NOTE: get_lr could be called multiple times in a list comprehensive fashion in LrUpdateHook
        if not self._is_check_timing() or self.current_lr == self.min_lr or self.bad_count_iter == self.current_iter:
            return self.current_lr

        if self.current_score is not None:
            score = self.current_score
        else:
            return self.current_lr

        if self.compare_func(score, self.best_score):
            self.best_score = score
            self.bad_count = 0
            self.last_iter = self.current_iter
        else:
            self.bad_count_iter = self.current_iter
            self.bad_count += 1

        logger.info(
            f"\nBest Score: {self.best_score}, Current Score: {score}, Patience: {self.patience} "
            f"Count: {self.bad_count}"
        )

        if self.bad_count >= self.patience:
            if self.current_iter - self.last_iter < self.iteration_patience:
                logger.info(
                    f"\nSkip LR dropping. Accumulated iteration "
                    f"{self.current_iter - self.last_iter} from the last "
                    f"improvement must be larger than {self.iteration_patience} to trigger "
                    f"LR dropping.",
                )
                return self.current_lr

            self.last_iter = self.current_iter
            self.bad_count = 0
            logger.info(
                f"\nDrop LR from: {self.current_lr}, to: " f"{max(self.current_lr * self.factor, self.min_lr)}"
            )
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
        return self.current_lr

    def before_run(self):
        """Called before_run in ReduceLROnPlateauLrUpdaterHook."""
        # TODO: remove overloaded method after fixing the issue
        #  https://github.com/open-mmlab/mmdetection/issues/6572
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lr = [group["initial_lr"] for group in self.optimizer.param_groups]
        self.bad_count = 0
        self.last_iter = 0
        self.current_lr = -1.0
        self.best_score = self.init_value_map[self.rule]

    def _set_lr(self, lr_groups):
        if isinstance(self.optimizer, dict):
            for k, optim in self.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(self.optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self):
        if isinstance(self.optimizer, dict):
            lr_groups = {}
            for k in self.optimizer.keys():
                _lr_group = [
                    self.get_lr(_base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(_base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters: int):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_train_epoch(self):
        if self.warmup_iters is None:
            self.warmup_iters = self.warmup_epochs

        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr()
        self._set_lr(self.regular_lr)

    def before_train_iter(self):
        cur_iter = self.current_iter
        assert isinstance(self.warmup_iters, int)
        if self.warmup is None or cur_iter > self.warmup_iters:
            return
        elif cur_iter == self.warmup_iters:
            self._set_lr(self.regular_lr)
        else:
            warmup_lr = self.get_warmup_lr(cur_iter)
            self._set_lr(warmup_lr)
