"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.adaptive_training_hooks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import LrUpdaterHook
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.evaluation import EvalHook
from mmcv.utils import Config

from otx.algorithms.common.adapters.mmcv.hooks.adaptive_training_hook import (
    AdaptiveTrainSchedulingHook,
)
from otx.algorithms.common.adapters.mmcv.hooks.early_stopping_hook import (
    EarlyStoppingHook,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockEvalHook(EvalHook):
    def __init__(self):
        self.interval = 5
        self.start = 5
        self.by_epoch = False


class MockLrUpdaterHook(LrUpdaterHook):
    def __init__(self):
        self.interval = 1
        self.patience = 1


class MockEarlyStoppingHook(EarlyStoppingHook):
    def __init__(self):
        self.start = 1
        self.interval = 1
        self.patience = 1


class MockCheckpointHook(CheckpointHook):
    def __init__(self):
        self.interval = 1
        self.by_epoch = False


class TestAdaptiveTrainSchedulingHook:
    """Test class for AdaptiveTrainSchedulingHook."""

    @e2e_pytest_unit
    def test_before_run(self) -> None:
        """Test before_run function."""

        eval_hook = MockEvalHook()
        mock_runner = Config({"hooks": [None, eval_hook]})
        hook = AdaptiveTrainSchedulingHook(enable_eval_before_run=True)
        hook.before_run(mock_runner)
        assert eval_hook.interval == 1
        assert eval_hook.start == 0
        assert hook._original_interval == 5

    @e2e_pytest_unit
    def test_before_train_iter(self) -> None:
        """Test before_train_iter function."""

        hook = AdaptiveTrainSchedulingHook(enable_eval_before_run=True, enable_adaptive_interval_hook=True)
        hook._original_interval = 5

        eval_hook = MockEvalHook()
        lr_hook = MockLrUpdaterHook()
        early_hook = MockEarlyStoppingHook()
        ckpt_hook = MockCheckpointHook()

        mock_runner = Config(
            {
                "hooks": [eval_hook, lr_hook, early_hook, ckpt_hook],
                "max_epochs": 200,
                "max_iters": 200,
                "data_loader": range(10),
                "epoch": 5,
            }
        )

        hook.before_train_iter(mock_runner)
        assert hook._initialized is True
        assert hook.max_interval == 5
        assert hook._original_interval is None
        assert eval_hook.interval == 4
        assert lr_hook.interval == 4
        assert lr_hook.patience == 2
        assert early_hook.interval == 4
        assert early_hook.patience == 3
        assert ckpt_hook.interval == 4
