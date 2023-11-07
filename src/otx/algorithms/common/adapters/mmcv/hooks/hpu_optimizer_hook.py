"""Optimizer hook for HPU."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import habana_frameworks.torch.core as htcore
from mmcv.runner import HOOKS, OptimizerHook


@HOOKS.register_module()
class HPUOptimizerHook(OptimizerHook):
    """A hook contains custom operations for the optimizer on HPU."""

    def after_train_iter(self, runner):
        """After train iter."""
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs["loss"], runner)
        runner.outputs["loss"].backward()
        htcore.mark_step()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({"grad_norm": float(grad_norm)}, runner.outputs["num_samples"])

        runner.optimizer.step()
        htcore.mark_step()
