"""Collections of hooks for common OTX algorithms."""

# Copyright (C) 2021-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import math
import os
from collections import defaultdict
from math import cos, inf, isnan, pi
from typing import Any, Dict, List, Optional

from mmcv.parallel import is_module_wrapper
from mmcv.runner import BaseRunner, EpochBasedRunner
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook, LoggerHook, LrUpdaterHook
from mmcv.utils import print_log

from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from otx.api.utils.argument_checks import check_input_parameters_type
from otx.mpa.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-instance-attributes, protected-access, too-many-arguments, unused-argument
@HOOKS.register_module()
class CancelTrainingHook(Hook):
    """CancelTrainingHook for Training Stopping."""

    @check_input_parameters_type()
    def __init__(self, interval: int = 5):
        """Periodically check whether whether a stop signal is sent to the runner during model training.

        Every 'check_interval' iterations, the work_dir for the runner is checked to see if a file '.stop_training'
        is present. If it is, training is stopped.

        :param interval: Period for checking for stop signal, given in iterations.

        """
        self.interval = interval

    @staticmethod
    def _check_for_stop_signal(runner: BaseRunner):
        """Log _check_for_stop_signal for CancelTrainingHook."""
        work_dir = runner.work_dir
        stop_filepath = os.path.join(work_dir, ".stop_training")
        if os.path.exists(stop_filepath):
            if isinstance(runner, EpochBasedRunner):
                epoch = runner.epoch
                runner._max_epochs = epoch  # Force runner to stop by pretending it has reached it's max_epoch
            runner.should_stop = True  # Set this flag to true to stop the current training epoch
            os.remove(stop_filepath)

    @check_input_parameters_type()
    def after_train_iter(self, runner: BaseRunner):
        """Log after_train_iter for CancelTrainingHook."""
        if not self.every_n_iters(runner, self.interval):
            return
        self._check_for_stop_signal(runner)


@HOOKS.register_module()
class EnsureCorrectBestCheckpointHook(Hook):
    """EnsureCorrectBestCheckpointHook.

    This hook makes sure that the 'best_mAP' checkpoint points properly to the best model, even if the best model is
    created in the last epoch.
    """

    @check_input_parameters_type()
    def after_run(self, runner: BaseRunner):
        """Called after train epoch hooks."""
        runner.call_hook("after_train_epoch")


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

    @check_input_parameters_type()
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
    @check_input_parameters_type()
    def log(self, runner: BaseRunner):
        """Log function for OTXLoggerHook."""
        tags = self.get_loggable_tags(runner, allow_text=False, tags_to_skip=())
        if runner.max_epochs is not None:
            normalized_iter = self.get_iter(runner) / runner.max_iters * runner.max_epochs
        else:
            normalized_iter = self.get_iter(runner)
        for tag, value in tags.items():
            curve = self.curves[tag]
            # Remove duplicates.
            if len(curve.x) > 0 and curve.x[-1] == normalized_iter:
                curve.x.pop()
                curve.y.pop()
            curve.x.append(normalized_iter)
            curve.y.append(value)

    @check_input_parameters_type()
    def after_train_epoch(self, runner: BaseRunner):
        """Called after_train_epoch in OTXLoggerHook."""
        # Iteration counter is increased right after the last iteration in the epoch,
        # temporarily decrease it back.
        runner._iter -= 1
        super().after_train_epoch(runner)
        runner._iter += 1


@HOOKS.register_module()
class OTXProgressHook(Hook):
    """OTXProgressHook for getting progress."""

    @check_input_parameters_type()
    def __init__(self, time_monitor: TimeMonitorCallback, verbose: bool = False):
        super().__init__()
        self.time_monitor = time_monitor
        self.verbose = verbose
        self.print_threshold = 1

    @check_input_parameters_type()
    def before_run(self, runner: BaseRunner):
        """Called before_run in OTXProgressHook."""
        total_epochs = runner.max_epochs if runner.max_epochs is not None else 1
        self.time_monitor.total_epochs = total_epochs
        self.time_monitor.train_steps = runner.max_iters // total_epochs if total_epochs else 1
        self.time_monitor.steps_per_epoch = self.time_monitor.train_steps + self.time_monitor.val_steps
        self.time_monitor.total_steps = max(math.ceil(self.time_monitor.steps_per_epoch * total_epochs), 1)
        self.time_monitor.current_step = 0
        self.time_monitor.current_epoch = 0
        self.time_monitor.on_train_begin()

    @check_input_parameters_type()
    def before_epoch(self, runner: BaseRunner):
        """Called before_epoch in OTXProgressHook."""
        self.time_monitor.on_epoch_begin(runner.epoch)

    @check_input_parameters_type()
    def after_epoch(self, runner: BaseRunner):
        """Called after_epoch in OTXProgressHook."""
        # put some runner's training status to use on the other hooks
        runner.log_buffer.output["current_iters"] = runner.iter
        self.time_monitor.on_epoch_end(runner.epoch, runner.log_buffer.output)

    @check_input_parameters_type()
    def before_iter(self, runner: BaseRunner):
        """Called before_iter in OTXProgressHook."""
        self.time_monitor.on_train_batch_begin(1)

    @check_input_parameters_type()
    def after_iter(self, runner: BaseRunner):
        """Called after_iter in OTXProgressHook."""
        # put some runner's training status to use on the other hooks
        runner.log_buffer.output["current_iters"] = runner.iter
        self.time_monitor.on_train_batch_end(1)
        if self.verbose:
            progress = self.progress
            if progress >= self.print_threshold:
                logger.warning(f"training progress {progress:.0f}%")
                self.print_threshold = (progress + 10) // 10 * 10

    @check_input_parameters_type()
    def before_val_iter(self, runner: BaseRunner):
        """Called before_val_iter in OTXProgressHook."""
        self.time_monitor.on_test_batch_begin(1, logger)

    @check_input_parameters_type()
    def after_val_iter(self, runner: BaseRunner):
        """Called after_val_iter in OTXProgressHook."""
        self.time_monitor.on_test_batch_end(1, logger)

    @check_input_parameters_type()
    def after_run(self, runner: BaseRunner):
        """Called after_run in OTXProgressHook."""
        self.time_monitor.on_train_end(1)
        if self.time_monitor.update_progress_callback:
            self.time_monitor.update_progress_callback(int(self.time_monitor.get_progress()))

    @property
    def progress(self):
        """Getting Progress from time monitor."""
        return self.time_monitor.get_progress()


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
    :param min_delta: Minimal decay applied to lr. If the difference between new and old lr is
                      smaller than eps, the update is ignored
    """

    rule_map = {"greater": lambda x, y: x > y, "less": lambda x, y: x < y}
    init_value_map = {"greater": -inf, "less": inf}
    greater_keys = [
        "acc",
        "top",
        "AR@",
        "auc",
        "precision",
        "mAP",
        "mDice",
        "mIoU",
        "mAcc",
        "aAcc",
    ]
    less_keys = ["loss"]

    @check_input_parameters_type()
    def __init__(
        self,
        interval: int,
        metric: str = "bbox_mAP",
        rule: Optional[str] = None,
        patience: int = 5,
        iteration_patience: int = 500,
        min_delta: float = 0.0,
    ):
        super().__init__()
        self.patience = patience
        self.iteration_patience = iteration_patience
        self.interval = interval
        self.min_delta = min_delta
        self._init_rule(rule, metric)

        self.min_delta *= 1 if self.rule == "greater" else -1
        self.last_iter = 0
        self.wait_count = 0
        self.by_epoch = True
        self.warmup_iters = 0
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

    @check_input_parameters_type()
    def before_run(self, runner: BaseRunner):
        """Called before_run in EarlyStoppingHook."""
        self.by_epoch = runner.max_epochs is not None
        for hook in runner.hooks:
            if isinstance(hook, LrUpdaterHook):
                self.warmup_iters = hook.warmup_iters
                break

    @check_input_parameters_type()
    def after_train_iter(self, runner: BaseRunner):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch:
            self._do_check_stopping(runner)

    @check_input_parameters_type()
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
            if self.compare_func(key_score - self.min_delta, self.best_score):
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

    rule_map = {"greater": lambda x, y: x > y, "less": lambda x, y: x < y}
    init_value_map = {"greater": -inf, "less": inf}
    greater_keys = [
        "acc",
        "top",
        "AR@",
        "auc",
        "precision",
        "mAP",
        "mDice",
        "mIoU",
        "mAcc",
        "aAcc",
    ]
    less_keys = ["loss"]

    @check_input_parameters_type()
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
        self.last_iter = 0
        self.current_lr = -1.0
        self.base_lr = []  # type: List
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

    @check_input_parameters_type()
    def get_lr(self, runner: BaseRunner, base_lr: float):
        """Called get_lr in ReduceLROnPlateauLrUpdaterHook."""
        if self.current_lr < 0:
            self.current_lr = base_lr

        if not self._is_check_timing(runner):
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

    @check_input_parameters_type()
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

    @check_input_parameters_type()
    def after_train_iter(self, runner: BaseRunner):
        """Called after_train_iter in StopLossNanTrainingHook."""
        if isnan(runner.outputs["loss"].item()):
            logger.warning("Early Stopping since loss is NaN")
            runner.should_stop = True


@HOOKS.register_module()
class EMAMomentumUpdateHook(Hook):
    """Exponential moving average (EMA) momentum update hook for self-supervised methods.

    This hook includes momentum adjustment in self-supervised methods following:
        m = 1 - ( 1- m_0) * (cos(pi * k / K) + 1) / 2,
        k: current step, K: total steps.

    :param end_momentum: The final momentum coefficient for the target network, defaults to 1.
    :param update_interval: Interval to update new momentum, defaults to 1.
    :param by_epoch: Whether updating momentum by epoch or not, defaults to False.
    """

    def __init__(self, end_momentum: float = 1.0, update_interval: int = 1, by_epoch: bool = False, **kwargs):
        self.by_epoch = by_epoch
        self.end_momentum = end_momentum
        self.update_interval = update_interval

    def before_train_epoch(self, runner: BaseRunner):
        """Called before_train_epoch in EMAMomentumUpdateHook."""
        if not self.by_epoch:
            return

        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        if not hasattr(model, "momentum"):
            raise AttributeError('The model must have attribute "momentum".')
        if not hasattr(model, "base_momentum"):
            raise AttributeError('The model must have attribute "base_momentum".')

        if self.every_n_epochs(runner, self.update_interval):
            cur_epoch = runner.epoch
            max_epoch = runner.max_epochs
            base_m = model.base_momentum
            updated_m = (
                self.end_momentum - (self.end_momentum - base_m) * (cos(pi * cur_epoch / float(max_epoch)) + 1) / 2
            )
            model.momentum = updated_m

    def before_train_iter(self, runner: BaseRunner):
        """Called before_train_iter in EMAMomentumUpdateHook."""
        if self.by_epoch:
            return

        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        if not hasattr(model, "momentum"):
            raise AttributeError('The model must have attribute "momentum".')
        if not hasattr(model, "base_momentum"):
            raise AttributeError('The model must have attribute "base_momentum".')

        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            max_iter = runner.max_iters
            base_m = model.base_momentum
            updated_m = (
                self.end_momentum - (self.end_momentum - base_m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2
            )
            model.momentum = updated_m

    def after_train_iter(self, runner: BaseRunner):
        """Called after_train_iter in EMAMomentumUpdateHook."""
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                runner.model.module.momentum_update()
            else:
                runner.model.momentum_update()


@HOOKS.register_module()
class ForceTrainModeHook(Hook):
    """Force train mode for model.

    This is a workaround of a bug in EvalHook from MMCV.
    If a model evaluation is enabled before training by setting 'start=0' in EvalHook,
    EvalHook does not put a model in a training mode again after evaluation.

    This simple hook forces to put a model in a training mode before every train epoch
    with the lowest priority.
    """

    def before_train_epoch(self, runner):
        """Make sure to put a model in a training mode before train epoch."""
        runner.model.train()


@HOOKS.register_module()
class TwoCropTransformHook(Hook):
    """TwoCropTransformHook with every specific interval.

    This hook decides whether using single pipeline or two pipelines
    implemented in `TwoCropTransform` for the current iteration.

    Args:
        interval (int): If `interval` == 1, both pipelines is used.
            If `interval` > 1, the first pipeline is used and then
            both pipelines are used every `interval`. Defaults to 1.
        by_epoch (bool): (TODO) Use `interval` by epoch. Defaults to False.
    """

    @check_input_parameters_type()
    def __init__(self, interval: int = 1, by_epoch: bool = False):
        assert interval > 0, f"interval (={interval}) must be positive value."
        if by_epoch:
            raise NotImplementedError("by_epoch is not implemented.")

        self.interval = interval
        self.cnt = 0

    @check_input_parameters_type()
    def _get_dataset(self, runner: BaseRunner):
        """Get dataset to handle `is_both`."""
        if hasattr(runner.data_loader.dataset, "dataset"):
            # for RepeatDataset
            dataset = runner.data_loader.dataset.dataset
        else:
            dataset = runner.data_loader.dataset

        return dataset

    # pylint: disable=inconsistent-return-statements
    @check_input_parameters_type()
    def _find_two_crop_transform(self, transforms: List[object]):
        """Find TwoCropTransform among transforms."""
        for transform in transforms:
            if transform.__class__.__name__ == "TwoCropTransform":
                return transform

    @check_input_parameters_type()
    def before_train_epoch(self, runner: BaseRunner):
        """Called before_train_epoch in TwoCropTransformHook."""
        # Always keep `TwoCropTransform` enabled.
        if self.interval == 1:
            return

        dataset = self._get_dataset(runner)
        two_crop_transform = self._find_two_crop_transform(dataset.pipeline.transforms)
        if self.cnt == self.interval - 1:
            # start using both pipelines
            two_crop_transform.is_both = True
        else:
            two_crop_transform.is_both = False

    @check_input_parameters_type()
    def after_train_iter(self, runner: BaseRunner):
        """Called after_train_iter in TwoCropTransformHook."""
        # Always keep `TwoCropTransform` enabled.
        if self.interval == 1:
            return

        if self.cnt < self.interval - 1:
            # Instead of using `runner.every_n_iters` or `runner.every_n_inner_iters`,
            # this condition is used to compare `self.cnt` with `self.interval` throughout the entire epochs.
            self.cnt += 1

        if self.cnt == self.interval - 1:
            dataset = self._get_dataset(runner)
            two_crop_transform = self._find_two_crop_transform(dataset.pipeline.transforms)
            if not two_crop_transform.is_both:
                # If `self.cnt` == `self.interval`-1, there are two cases,
                # 1. `self.cnt` was updated in L709, so `is_both` must be on for the next iter.
                # 2. if the current iter was already conducted, `is_both` must be off.
                two_crop_transform.is_both = True
            else:
                two_crop_transform.is_both = False
                self.cnt = 0
