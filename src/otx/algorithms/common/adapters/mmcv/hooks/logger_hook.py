"""Logger hooks."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Any, Dict, Optional

from mmcv.runner import BaseRunner
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook, LoggerHook

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class OTXLoggerHook(LoggerHook):
    """OTXLoggerHook for Logging."""

    class Curve:
        """Curve with x (epochs) & y (scores)."""

        def __init__(self):
            self.x = []
            self.y = []

        def __repr__(self):
            """Repr function."""
            points = []
            for x, y in zip(self.x, self.y):
                points.append(f"({x},{y})")
            return "curve[" + ",".join(points) + "]"

    _TAGS_TO_SKIP = (
        "accuracy_top-1",
        "current_iters",
        "decode.acc_seg",
        "decode.loss_ce_ignore",
    )

    _TAGS_TO_RENAME = {
        "train/time": "train/time (sec/iter)",
        "train/data_time": "train/data_time (sec/iter)",
        "val/accuracy": "val/accuracy (%)",
    }

    def __init__(
        self,
        curves: Optional[Dict[Any, Curve]] = None,
        interval: int = 10,
        ignore_last: bool = True,
        reset_flag: bool = True,
        by_epoch: bool = True,
    ):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.curves = curves if curves is not None else defaultdict(self.Curve)

    @master_only
    def log(self, runner: BaseRunner):
        """Log function for OTXLoggerHook."""
        tags = self.get_loggable_tags(runner, allow_text=False, tags_to_skip=self._TAGS_TO_SKIP)
        if runner.max_epochs is not None:
            normalized_iter = self.get_iter(runner) / runner.max_iters * runner.max_epochs
        else:
            normalized_iter = self.get_iter(runner)
        for tag, value in tags.items():
            tag = self._TAGS_TO_RENAME.get(tag, tag)
            curve = self.curves[tag]
            # Remove duplicates.
            if len(curve.x) > 0 and curve.x[-1] == normalized_iter:
                curve.x.pop()
                curve.y.pop()
            curve.x.append(normalized_iter)
            curve.y.append(value)

    def before_run(self, runner: BaseRunner):
        """Called before_run in OTXLoggerHook."""
        super().before_run(runner)
        self.curves.clear()

    def after_train_epoch(self, runner: BaseRunner):
        """Called after_train_epoch in OTXLoggerHook."""
        # Iteration counter is increased right after the last iteration in the epoch,
        # temporarily decrease it back.
        runner._iter -= 1
        super().after_train_epoch(runner)
        runner._iter += 1


@HOOKS.register_module()
class LoggerReplaceHook(Hook):
    """replace logger in the runner to the OTX logger.

    DO NOT INCLUDE this hook to the recipe directly.
    OTX will add this hook to all recipe internally.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_run(self, runner):
        """Replace logger."""
        runner.logger = logger
        logger.info("logger in the runner is replaced to the OTX logger")
