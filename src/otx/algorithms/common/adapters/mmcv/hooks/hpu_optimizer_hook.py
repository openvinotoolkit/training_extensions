"""Optimizer hook for HPU."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, OptimizerHook, Fp16OptimizerHook, allreduce_grads
from mmcls.core import DistOptimizerHook
import habana_frameworks.torch.core as htcore


@HOOKS.register_module()
class HPUOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        
        # API call to trigger execution
        htcore.mark_step()

        runner.optimizer.step()
        
        # API call to trigger execution
        htcore.mark_step()


@HOOKS.register_module()
class HPUFp16OptimizerHook(Fp16OptimizerHook):
    def after_train_iter(self, runner) -> None:
            """Backward optimization steps for Mixed Precision Training. For
            dynamic loss scaling, please refer `loss_scalar.py`

            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients (fp16).
            3. Copy gradients from the model to the fp32 weight copy.
            4. Scale the gradients back and update the fp32 weight copy.
            5. Copy back the params from fp32 weight copy to the fp16 model.
            6. Save loss_scaler state_dict for resume purpose.
            """
            # clear grads of last iteration
            runner.model.zero_grad()
            runner.optimizer.zero_grad()
            # scale the loss value
            scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
            scaled_loss.backward()
            # copy fp16 grads in the model to fp32 params in the optimizer

            fp32_weights = []
            for param_group in runner.optimizer.param_groups:
                fp32_weights += param_group['params']
            self.copy_grads_to_fp32(runner.model, fp32_weights)
            # allreduce grads
            if self.distributed:
                allreduce_grads(fp32_weights, self.coalesce,
                                self.bucket_size_mb)

            has_overflow = self.loss_scaler.has_overflow(fp32_weights)
            # if has overflow, skip this iteration
            if not has_overflow:
                # scale the gradients back
                for param in fp32_weights:
                    if param.grad is not None:
                        param.grad.div_(self.loss_scaler.loss_scale)
                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(fp32_weights)
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update(
                            {'grad_norm': float(grad_norm)},
                            runner.outputs['num_samples'])
                        
                # API call to trigger execution
                htcore.mark_step()
                
                # update fp32 params
                runner.optimizer.step()
                
                # API call to trigger execution
                htcore.mark_step()

                # copy fp32 params to the fp16 model
                self.copy_params_to_fp16(runner.model, fp32_weights)
            self.loss_scaler.update_scale(has_overflow)
            if has_overflow:
                runner.logger.warning('Check overflow, downscale loss scale '
                                      f'to {self.loss_scaler.cur_scale}')

            # save state_dict of loss_scaler
            runner.meta.setdefault(
                'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()
    
    
@HOOKS.register_module()
class HPUDistOptimizerHook(DistOptimizerHook):
    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
            
        # API call to trigger execution
        htcore.mark_step()
        
        runner.optimizer.step()
        
        # API call to trigger execution
        htcore.mark_step()
