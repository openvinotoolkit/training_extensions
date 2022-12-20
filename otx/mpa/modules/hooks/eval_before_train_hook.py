# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS
from mmcv.runner import EvalHook as SegEvalHook
from mmcv.runner import Hook
from mmdet.core.evaluation.eval_hooks import EvalHook as DetEvalHook
from mpa.modules.hooks.checkpoint_hook import CheckpointHookWithValResults
from mpa.modules.hooks.eval_hook import CustomEvalHook as ClsEvalHook


@HOOKS.register_module()
class EvalBeforeTrainHook(Hook):
    """Hook to evaluate and save model weight before training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._executed = False

    def before_train_epoch(self, runner):
        """Execute the evaluation hook before training"""
        if not self._executed:
            for hook in runner.hooks:
                if self.check_eval_hook(hook):
                    self.execute_hook(hook, runner)
                    if not issubclass(type(hook), ClsEvalHook):
                        break

                # cls task saves the model weight in a checkpoint hook after eval hook is executed
                if issubclass(type(hook), CheckpointHookWithValResults):
                    self.execute_hook(hook, runner)
                    break

            self._executed = True

    @staticmethod
    def check_eval_hook(hook: Hook):
        """Check that the hook is an evaluation hook."""
        return issubclass(type(hook), (ClsEvalHook, DetEvalHook, SegEvalHook))

    @staticmethod
    def execute_hook(hook: Hook, runner):
        """Execute after_train_epoch or iter depending on `by_epoch` value"""
        if getattr(hook, "by_epoch", True):
            hook.after_train_epoch(runner)
        else:
            hook.after_train_iter(runner)
