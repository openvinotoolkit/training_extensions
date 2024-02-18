"""Module for Sharpness-aware Minimization optimizer hook implementation for MMCV Runners with FP16 precision."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.runner import HOOKS, Fp16OptimizerHook


@HOOKS.register_module()
class Fp16SAMOptimizerHook(Fp16OptimizerHook):
    """Sharpness-aware Minimization optimizer hook.

    Implemented as OptimizerHook for MMCV Runners
    - Paper ref: https://arxiv.org/abs/2010.01412
    - code ref: https://github.com/davda54/sam
    """

    def __init__(self, rho=0.05, start_epoch=1, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho
        self.start_epoch = start_epoch
        if rho < 0.0:
            raise ValueError("rho should be greater than 0 for SAM optimizer")

    def after_train_iter(self, runner):
        """Perform SAM optimization.

        0. compute current loss (DONE IN model.train_step())
        1. compute current gradient
        2. move param to the approximate local maximum: w + e(w) = w + rho*norm_grad
        3. compute maximum loss
        4. compute SAM gradient on maximum loss
        5. restore parram to original param
        6. update param using SAM gradient

        Assuming model.current_batch had been set in model.train_step()
        """
        current_batch = self._get_current_batch(runner.model)
        if current_batch is None or runner.epoch + 1 < self.start_epoch:
            # Apply original parameter update
            return super().after_train_iter(runner)

        # Current gradient
        runner.optimizer.zero_grad()
        curr_loss = runner.outputs["loss"]
        curr_loss.backward()

        # Move to local maximum
        param2move = {}
        with torch.no_grad():
            scale = self.rho / (self._grad_norm(runner.optimizer) + 1e-12)
            for param_group in runner.optimizer.param_groups:
                for param in param_group["params"]:
                    if param.grad is None:
                        continue
                    e_w = param.grad * scale.to(param)
                    param.add_(e_w)  # Climb toward gradient (increasing loss)
                    param2move[param] = e_w  # Saving for param restore

        # SAM gradient
        runner.optimizer.zero_grad()
        max_outputs = runner.model.train_step(current_batch, runner.optimizer)  # forward() on maxima param
        max_loss = max_outputs["loss"]
        self.loss_scaler.scale(max_loss).backward()
        self.loss_scaler.unscale_(runner.optimizer)

        # Restore to original param
        with torch.no_grad():
            for param_group in runner.optimizer.param_groups:
                for param in param_group["params"]:
                    if param.grad is None:
                        continue
                    param.sub_(param2move[param])  # Down to original param

        # Shaprpness-aware param update
        self.loss_scaler.step(runner.optimizer)  # param -= lr * sam_grad
        self.loss_scaler.update(self._scale_update_param)

        runner.meta.setdefault("fp16", {})["loss_scaler"] = self.loss_scaler.state_dict()
        runner.log_buffer.update({"sharpness": float(max_loss - curr_loss), "max_loss": float(max_loss)})
        return None

    def _get_current_batch(self, model):
        if hasattr(model, "module"):
            model = model.module
        return getattr(model, "current_batch", None)

    def _grad_norm(self, optimizer):
        # put everything on the same device, in case of model parallelism
        shared_device = optimizer.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in optimizer.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm
