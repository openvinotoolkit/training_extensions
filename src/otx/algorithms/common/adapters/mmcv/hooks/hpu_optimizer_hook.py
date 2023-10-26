"""Optimizer hook for HPU."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, OptimizerHook
from mmcls.core import DistOptimizerHook
import habana_frameworks.torch.core as htcore


@HOOKS.register_module()
class HPUOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'].backward()
        htcore.mark_step()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        
        runner.optimizer.step()
        htcore.mark_step()
    
    
@HOOKS.register_module()
class HPUDistOptimizerHook(DistOptimizerHook):
    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        htcore.mark_step()

        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
            
        runner.optimizer.step()
        htcore.mark_step()
