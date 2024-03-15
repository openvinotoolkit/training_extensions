"""Module for definig CustomEvalHook for classification task."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from os import path as osp

import mmcv
import torch
from mmcv.runner import HOOKS, EvalHook
from torch.utils.data import DataLoader


@HOOKS.register_module()
class CustomEvalHook(EvalHook):
    """Custom Evaluation hook for the OTX.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(
        self,
        *args,
        ema_eval_start_epoch=10,
        **kwargs,
    ):
        metric = kwargs["metric"]
        self.metric = None
        if isinstance(metric, str):
            self.metric = "top-1" if metric == "accuracy" else metric
        else:
            self.metric = metric[0]
            if metric.count("class_accuracy") > 0:
                self.metric = "accuracy"
            elif metric.count("accuracy") > 0:
                self.metric = "top-1"
        super().__init__(*args, **kwargs, save_best=self.metric, rule="greater")
        self.ema_eval_start_epoch = ema_eval_start_epoch

        self.best_loss = 9999999.0
        self.best_score = 0.0
        self.save_mode = self.eval_kwargs.get("save_mode", "score")

    def _do_evaluate(self, runner, ema=False):
        """Perform evaluation."""
        results = single_gpu_test(runner.model, self.dataloader)
        if ema and hasattr(runner, "ema_model") and (runner.epoch >= self.ema_eval_start_epoch):
            results_ema = single_gpu_test(runner.ema_model.module, self.dataloader)
            self.evaluate(runner, results, results_ema)
        else:
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """Check whether current epoch is to be evaluated or not."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        self._do_evaluate(runner, ema=True)

    def after_train_iter(self, runner):
        """Check whether current iteration is to be evaluated or not."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        self._do_evaluate(runner)

    def evaluate(self, runner, results, results_ema=None):
        """Evaluate predictions from model with ground truth."""
        eval_res = self.dataloader.dataset.evaluate(results, logger=runner.logger, **self.eval_kwargs)
        score = eval_res[self.metric]
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val

        if results_ema:
            eval_res_ema = self.dataloader.dataset.evaluate(results_ema, logger=runner.logger, **self.eval_kwargs)
            score_ema = eval_res_ema[self.metric]
            for name, val in eval_res_ema.items():
                runner.log_buffer.output[name + "_EMA"] = val
            if score_ema > score:
                runner.save_ema_model = True

        runner.log_buffer.ready = True
        if score >= self.best_score:
            self.best_score = score
            runner.save_ckpt = True


def single_gpu_test(model, data_loader):
    """Single gpu test for inference."""
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        batch_size = data["img"].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    prog_bar.file.write("\n")
    return results


@HOOKS.register_module()
class DistCustomEvalHook(CustomEvalHook):
    """Distributed Custom Evaluation Hook for Multi-GPU environment."""

    def __init__(self, dataloader, interval=1, gpu_collect=False, by_epoch=True, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError("dataloader must be a pytorch DataLoader, but got " f"{type(dataloader)}")
        self.gpu_collect = gpu_collect
        super().__init__(dataloader, interval, by_epoch=by_epoch, **eval_kwargs)

    def _do_evaluate(self, runner):
        """Perform evaluation."""
        from mmcls.apis import multi_gpu_test

        results = multi_gpu_test(
            runner.model, self.dataloader, tmpdir=osp.join(runner.work_dir, ".eval_hook"), gpu_collect=self.gpu_collect
        )
        if runner.rank == 0:
            print("\n")
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        """Check whether current epoch is to be evaluated or not."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        self._do_evaluate(runner)

    def after_train_iter(self, runner):
        """Check whether current iteration is to be evaluated or not."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        self._do_evaluate(runner)
