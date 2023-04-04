"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.early_stopping_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from logging import Logger
from math import inf

import numpy as np
import pytest
from mmcv.runner import BaseRunner, LrUpdaterHook
from mmcv.utils import Config

from otx.algorithms.common.adapters.mmcv.hooks.early_stopping_hook import (
    EarlyStoppingHook,
    LazyEarlyStoppingHook,
    ReduceLROnPlateauLrUpdaterHook,
    StopLossNanTrainingHook,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockRunner(BaseRunner):
    """Mock class for BaseRunner."""

    def __init__(self) -> None:
        self._max_epochs = None
        self._iter = 9
        self._hooks = [LrUpdaterHook(warmup_iters=3)]
        self._rank = 0
        self.log_buffer = Config({"output": {"acc": 1.0}})
        self.logger = Logger("otx")
        self.should_stop = False
        self.optimizer = Config({"param_groups": [{"lr": 1e-4}]})
        self.bbox_mAP = 50.0

    def run(self):
        pass

    def save_checkpoint(self):
        pass

    def train(self):
        pass

    def val(self):
        pass


class TestEarlyStoppingHook:
    """Test class for EarlyStoppingHook."""

    @e2e_pytest_unit
    def test_init_rule(self) -> None:
        """Test funciton for init_rule function."""

        hook = EarlyStoppingHook(interval=5)
        with pytest.raises(KeyError):
            hook._init_rule("Invalid Key", "Invalid Indicator")
        with pytest.raises(ValueError):
            hook._init_rule(None, "Invalid Indicator")
        hook._init_rule("greater", "acc")
        assert hook.rule == "greater"
        assert hook.key_indicator == "acc"
        assert hook.compare_func(5, 9) is False
        hook._init_rule("less", "loss")
        assert hook.rule == "less"
        assert hook.key_indicator == "loss"
        assert hook.compare_func(5, 9) is True

    @e2e_pytest_unit
    def test_before_run(self) -> None:
        """Test function for before_run."""

        hook = EarlyStoppingHook(interval=5)
        runner = MockRunner()
        hook.before_run(runner)
        assert hook.by_epoch is False
        assert hook.warmup_iters == 3

        runner._hooks = []
        hook = EarlyStoppingHook(interval=5)
        with pytest.raises(ValueError):
            hook.before_run(runner)

    @e2e_pytest_unit
    def test_after_train_iter(self, mocker) -> None:
        """Test after_train_iter function."""

        mocker.patch.object(EarlyStoppingHook, "_do_check_stopping", return_value=True)
        hook = EarlyStoppingHook(interval=5)
        runner = MockRunner()
        hook.by_epoch = False
        hook.after_train_iter(runner)

    @e2e_pytest_unit
    def test_after_train_epoch(self, mocker) -> None:
        """Test after_train_epoch function."""

        mocker.patch.object(EarlyStoppingHook, "_do_check_stopping", return_value=True)
        runner = MockRunner()
        hook = EarlyStoppingHook(interval=5)
        hook.by_epoch = True
        hook.after_train_epoch(runner)

    @e2e_pytest_unit
    def test_do_check_stopping(self, mocker):
        """Test _do_check_stopping function."""

        runner = MockRunner()
        mocker.patch.object(EarlyStoppingHook, "_should_check_stopping", return_value=False)
        hook = EarlyStoppingHook(interval=5)
        hook.warmup_iters = 3
        hook._do_check_stopping(runner)
        mocker.patch.object(EarlyStoppingHook, "_should_check_stopping", return_value=True)
        runner.log_buffer = Config({"output": []})
        with pytest.raises(KeyError):
            hook._do_check_stopping(runner)
        runner = MockRunner()
        hook.key_indicator = "acc"
        hook._do_check_stopping(runner)
        assert hook.best_score == 1.0
        assert hook.wait_count == 0
        assert hook.last_iter == 9

        hook = EarlyStoppingHook(interval=5)
        hook.by_epoch = False
        hook.key_indicator = "acc"
        hook.warmup_iters = 3
        hook.best_score = 2.0
        hook.patience = 1.0
        hook.iteration_patience = 0.0
        hook._do_check_stopping(runner)
        assert hook.wait_count == 1
        assert runner.should_stop is True

        runner.should_stop = False
        hook = EarlyStoppingHook(interval=5)
        hook.by_epoch = False
        hook.key_indicator = "acc"
        hook.warmup_iters = 3
        hook.best_score = 2.0
        hook.patience = 1.0
        hook.iteration_patience = 20.0
        hook._do_check_stopping(runner)
        assert hook.wait_count == 1
        assert runner.should_stop is False

    @e2e_pytest_unit
    def test_should_check_stopping(self) -> None:
        """Test _should_check_stopping function."""

        hook = EarlyStoppingHook(interval=5)
        hook.by_epoch = False
        runner = MockRunner()
        assert hook._should_check_stopping(runner) is True

        runner._iter = 8
        assert hook._should_check_stopping(runner) is False


class TestLazyEarlyStoppingHook:
    """Test LazyEarlyStoppingHook function."""

    def test_should_check_stopping(self) -> None:
        hook = LazyEarlyStoppingHook(interval=5)
        hook.by_epoch = False

        runner = MockRunner()
        runner._iter = 8
        assert hook._should_check_stopping(runner) is False

        hook.start = 20
        assert hook._should_check_stopping(runner) is False

        runner._iter = 17
        hook.start = 4
        assert hook._should_check_stopping(runner) is False

        runner._iter = 18
        hook.start = 4
        assert hook._should_check_stopping(runner) is True


class TestReduceLROnPlateauLrUpdaterHook:
    """Test class for ReduceLROnPlateauLrUpdaterHook."""

    @e2e_pytest_unit
    def test_init_rule(self) -> None:
        """Test funciton for init_rule function."""

        hook = ReduceLROnPlateauLrUpdaterHook(interval=5, min_lr=1e-5)
        with pytest.raises(KeyError):
            hook._init_rule("Invalid Key", "Invalid Indicator")
        with pytest.raises(ValueError):
            hook._init_rule(None, "Invalid Indicator")
        hook._init_rule("greater", "acc")
        assert hook.rule == "greater"
        assert hook.key_indicator == "acc"
        assert hook.compare_func(5, 9) is False
        hook._init_rule("less", "loss")
        assert hook.rule == "less"
        assert hook.key_indicator == "loss"
        assert hook.compare_func(5, 9) is True

    @e2e_pytest_unit
    def test_is_check_timing(self) -> None:
        """Test _should_check_stopping function."""

        hook = ReduceLROnPlateauLrUpdaterHook(interval=5, min_lr=1e-5)
        hook.by_epoch = False
        runner = MockRunner()
        assert hook._is_check_timing(runner) is False

    @e2e_pytest_unit
    def test_get_lr(self, mocker) -> None:
        """Test function for get_lr."""

        mocker.patch.object(ReduceLROnPlateauLrUpdaterHook, "_is_check_timing", return_value=False)
        hook = ReduceLROnPlateauLrUpdaterHook(interval=5, min_lr=1e-5)
        hook.warmup_iters = 3
        runner = MockRunner()
        assert hook.get_lr(runner, 1e-2) == 1e-2

        mocker.patch.object(ReduceLROnPlateauLrUpdaterHook, "_is_check_timing", return_value=True)
        hook = ReduceLROnPlateauLrUpdaterHook(interval=5, min_lr=1e-5)
        hook.warmup_iters = 3
        runner = MockRunner()
        assert hook.get_lr(runner, 1e-2) == 1e-2
        assert hook.bad_count == 0

        mocker.patch.object(ReduceLROnPlateauLrUpdaterHook, "_is_check_timing", return_value=True)
        hook = ReduceLROnPlateauLrUpdaterHook(interval=5, min_lr=1e-5)
        hook.best_score = 90
        hook.warmup_iters = 3
        hook.bad_count = 2
        hook.iteration_patience = 5
        hook.last_iter = 8
        runner = MockRunner()
        assert hook.get_lr(runner, 1e-2) == 1e-2

        hook = ReduceLROnPlateauLrUpdaterHook(interval=5, min_lr=1e-5)
        hook.best_score = 90
        hook.warmup_iters = 3
        hook.bad_count = 2
        hook.iteration_patience = 5
        hook.last_iter = 2
        runner = MockRunner()
        assert hook.get_lr(runner, 1e-3) == 1e-3
        assert hook.last_iter == 2
        assert hook.bad_count == 2

    @e2e_pytest_unit
    def test_before_run(self) -> None:
        """Test function for before_run."""

        hook = ReduceLROnPlateauLrUpdaterHook(interval=5, min_lr=1e-5)
        runner = MockRunner()
        hook.before_run(runner)
        assert hook.base_lr == [1e-4]
        assert hook.bad_count == 0
        assert hook.last_iter == 0
        assert hook.current_lr == -1.0
        assert hook.best_score == -inf


class TestStopLossNanTrainingHook:
    """Test class for StopLossNanTrainingHook."""

    def test_after_train_iter(self) -> None:
        hook = StopLossNanTrainingHook()
        runner = MockRunner()
        runner.outputs = {"loss": np.array([np.nan])}
        hook.after_train_iter(runner)
        assert runner.should_stop is True
