"""Eval Before Run hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook
from mmcv.runner.hooks.evaluation import EvalHook

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()

# pylint: disable=too-many-arguments, too-many-instance-attributes


@HOOKS.register_module()
class EvalBeforeRunHook(Hook):
    """Eval Before Run Hook.

    Enables the evaluation before the first run.

    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._original_interval = None

    def before_run(self, runner):
        """Before run."""
        hook = self.get_evalhook(runner)
        if hook is None:
            logger.warning("EvalHook is not found in runner. Skipping enabling evaluation before run.")
            return
        self._original_interval = hook.interval
        hook.interval = 1
        hook.start = 0

    def before_train_iter(self, runner):
        """Before train iter."""
        if self._original_interval is not None:
            hook = self.get_evalhook(runner)
            hook.interval = self._original_interval
            self._original_interval = None

    def get_evalhook(self, runner):
        """Get evaluation hook."""
        target_hook = None
        for hook in runner.hooks:
            if isinstance(hook, EvalHook):
                assert target_hook is None, "More than 1 EvalHook is found in runner."
                target_hook = hook
        return target_hook
