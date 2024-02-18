"""Early stopping hooks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from math import inf, isnan
from typing import List, Optional

from mmcv.runner import BaseRunner, LrUpdaterHook
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import print_log

from otx.utils.logger import get_logger

logger = get_logger()

# pylint: disable=too-many-arguments, too-many-instance-attributes


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Cancel training when a metric has stopped improving.

    Early Stopping hook monitors a metric quantity and if no improvement is seen for a ‘patience’
    number of epochs, the training is cancelled.

    :param interval: the number of intervals for checking early stop. The interval number should be
                     the same as the evaluation interval - the `interval` variable set in
                     `evaluation` config.
    :param metric: the metric name to be monitored
    :param rule: greater or less.  In `less` mode, training will stop when the metric has stopped
                 decreasing and in `greater` mode it will stop when the metric has stopped
                 increasing.
    :param patience: Number of epochs with no improvement after which the training will be reduced.
                     For example, if patience = 2, then we will ignore the first 2 epochs with no
                     improvement, and will only cancel the training after the 3rd epoch if the
                     metric still hasn’t improved then
    :param iteration_patience: Number of iterations must be trained after the last improvement
                               before training stops. The same as patience but the training
                               continues if the number of iteration is lower than iteration_patience
                               This variable makes sure a model is trained enough for some
                               iterations after the last improvement before stopping.
    :param min_delta_ratio: Minimal ratio value to check the best score. If the difference between current and
                      best score is smaller than (current_score * (1-min_delta_ratio)), best score will not be changed.
    """

    rule_map = {"greater": lambda x, y: x > y, "less": lambda x, y: x < y}
    init_value_map = {"greater": -inf, "less": inf}
    greater_keys = ["acc", "top", "AR@", "auc", "precision", "mAP", "mDice", "mIoU", "mAcc", "aAcc", "MHAcc"]
    less_keys = ["loss"]

    def __init__(
        self,
        interval: int,
        metric: str = "bbox_mAP",
        rule: Optional[str] = None,
        patience: int = 5,
        iteration_patience: int = 500,
        min_delta_ratio: float = 0.0,
    ):
        super().__init__()
        self.patience = patience
        self.iteration_patience = iteration_patience
        self.interval = interval
        self.min_delta_ratio = min_delta_ratio
        self._init_rule(rule, metric)

        self.min_delta_ratio *= 1 if self.rule == "greater" else -1
        self.last_iter = 0
        self.wait_count = 0
        self.best_score = self.init_value_map[self.rule]
        self.warmup_iters = None
        self.by_epoch = True

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

    def before_run(self, runner: BaseRunner):
        """Called before_run in EarlyStoppingHook."""
        if runner.max_epochs is None:
            self.by_epoch = False
        for hook in runner.hooks:
            if isinstance(hook, LrUpdaterHook):
                self.warmup_iters = hook.warmup_iters
                break
        if getattr(self, "warmup_iters", None) is None:
            raise ValueError("LrUpdaterHook must be registered to runner.")

    def after_train_iter(self, runner: BaseRunner):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch:
            self._do_check_stopping(runner)

    def after_train_epoch(self, runner: BaseRunner):
        """Called after every training epoch to evaluate the results."""
        if self.by_epoch:
            self._do_check_stopping(runner)

    def _do_check_stopping(self, runner):
        """Called _do_check_stopping in EarlyStoppingHook."""
        if not self._should_check_stopping(runner) or self.warmup_iters > runner.iter:
            return

        if runner.rank == 0:
            if self.key_indicator not in runner.log_buffer.output:
                raise KeyError(
                    f"metric {self.key_indicator} does not exist in buffer. Please check "
                    f"{self.key_indicator} is cached in evaluation output buffer"
                )

            key_score = runner.log_buffer.output[self.key_indicator]
            if self.compare_func(key_score - (key_score * self.min_delta_ratio), self.best_score):
                self.best_score = key_score
                self.wait_count = 0
                self.last_iter = runner.iter
            else:
                self.wait_count += 1
                if self.wait_count >= self.patience:
                    if runner.iter - self.last_iter < self.iteration_patience:
                        print_log(
                            f"\nSkip early stopping. Accumulated iteration "
                            f"{runner.iter - self.last_iter} from the last "
                            f"improvement must be larger than {self.iteration_patience} to trigger "
                            f"Early Stopping.",
                            logger=runner.logger,
                        )
                        return
                    stop_point = runner.epoch if self.by_epoch else runner.iter
                    print_log(
                        f"\nEarly Stopping at :{stop_point} with " f"best {self.key_indicator}: {self.best_score}",
                        logger=runner.logger,
                    )
                    runner.should_stop = True

    def _should_check_stopping(self, runner):
        """Called _should_check_stopping in EarlyStoppingHook."""
        check_time = self.every_n_epochs if self.by_epoch else self.every_n_iters
        if not check_time(runner, self.interval):
            # No evaluation during the interval.
            return False
        return True


@HOOKS.register_module()
class LazyEarlyStoppingHook(EarlyStoppingHook):
    """Lazy early stop hook."""

    def __init__(
        self,
        interval: int,
        metric: str = "bbox_mAP",
        rule: str = None,
        patience: int = 5,
        iteration_patience: int = 500,
        min_delta_ratio: float = 0.0,
        start: int = None,
    ):
        self.start = start
        super().__init__(interval, metric, rule, patience, iteration_patience, min_delta_ratio)

    def _should_check_stopping(self, runner):
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters

        if self.start is None:
            if not check_time(runner, self.interval):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            return False
        elif (current + 1 - self.start) % self.interval:
            return False
        return True


@HOOKS.register_module(force=True)
class ReduceLROnPlateauLrUpdaterHook(LrUpdaterHook):
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
        min_lr: float,
        interval: int,
        metric: str = "bbox_mAP",
        rule: Optional[str] = None,
        factor: float = 0.1,
        patience: int = 3,
        iteration_patience: int = 300,
        **kwargs,
    ):
        super().__init__(**kwargs)
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

    def _is_check_timing(self, runner: BaseRunner) -> bool:
        """Check whether current epoch or iter is multiple of self.interval, skip during warmup interations."""
        check_time = self.after_each_n_epochs if self.by_epoch else self.after_each_n_iters
        return check_time(runner, self.interval) and (self.warmup_iters <= runner.iter)

    def after_each_n_epochs(self, runner: BaseRunner, interval: int) -> bool:
        """Check whether current epoch is a next epoch after multiples of interval."""
        return runner.epoch % interval == 0 if interval > 0 and runner.epoch != 0 else False

    def after_each_n_iters(self, runner: BaseRunner, interval: int) -> bool:
        """Check whether current iter is a next iter after multiples of interval."""
        return runner.iter % interval == 0 if interval > 0 and runner.iter != 0 else False

    def get_lr(self, runner: BaseRunner, base_lr: float):
        """Called get_lr in ReduceLROnPlateauLrUpdaterHook."""
        if self.current_lr < 0:
            self.current_lr = base_lr

        # NOTE: get_lr could be called multiple times in a list comprehensive fashion in LrUpdateHook
        if not self._is_check_timing(runner) or self.current_lr == self.min_lr or self.bad_count_iter == runner.iter:
            return self.current_lr

        if hasattr(runner, "all_metrics"):
            score = runner.all_metrics.get(self.metric, 0.0)
        else:
            return self.current_lr

        if self.compare_func(score, self.best_score):
            self.best_score = score
            self.bad_count = 0
            self.last_iter = runner.iter
        else:
            self.bad_count_iter = runner.iter
            self.bad_count += 1

        print_log(
            f"\nBest Score: {self.best_score}, Current Score: {score}, Patience: {self.patience} "
            f"Count: {self.bad_count}",
            logger=runner.logger,
        )

        if self.bad_count >= self.patience:
            if runner.iter - self.last_iter < self.iteration_patience:
                print_log(
                    f"\nSkip LR dropping. Accumulated iteration "
                    f"{runner.iter - self.last_iter} from the last "
                    f"improvement must be larger than {self.iteration_patience} to trigger "
                    f"LR dropping.",
                    logger=runner.logger,
                )
                return self.current_lr

            self.last_iter = runner.iter
            self.bad_count = 0
            print_log(
                f"\nDrop LR from: {self.current_lr}, to: " f"{max(self.current_lr * self.factor, self.min_lr)}",
                logger=runner.logger,
            )
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
        return self.current_lr

    def before_run(self, runner: BaseRunner):
        """Called before_run in ReduceLROnPlateauLrUpdaterHook."""
        # TODO: remove overloaded method after fixing the issue
        #  https://github.com/open-mmlab/mmdetection/issues/6572
        for group in runner.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lr = [group["initial_lr"] for group in runner.optimizer.param_groups]
        self.bad_count = 0
        self.last_iter = 0
        self.current_lr = -1.0
        self.best_score = self.init_value_map[self.rule]


@HOOKS.register_module(force=True)
class StopLossNanTrainingHook(Hook):
    """StopLossNanTrainingHook."""

    def after_train_iter(self, runner: BaseRunner):
        """Called after_train_iter in StopLossNanTrainingHook."""
        if isnan(runner.outputs["loss"].item()):
            logger.warning("Early Stopping since loss is NaN")
            runner.should_stop = True
