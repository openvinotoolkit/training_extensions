"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.eval_before_run_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner.hooks.evaluation import EvalHook
from mmcv.utils import Config

from otx.algorithms.common.adapters.mmcv.hooks.eval_before_run_hook import (
    EvalBeforeRunHook,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockEvalHook(EvalHook):
    def __init__(self):
        self.interval = 5
        self.start = 5
        self.by_epoch = False


class TestEvalBeforeRunHook:
    """Test class for AdaptiveTrainSchedulingHook."""

    @e2e_pytest_unit
    def test_before_run(self) -> None:
        """Test before_run function."""

        eval_hook = MockEvalHook()
        mock_runner = Config({"hooks": [None, eval_hook]})
        hook = EvalBeforeRunHook()
        hook.before_run(mock_runner)
        assert eval_hook.interval == 1
        assert eval_hook.start == 0
        assert hook._original_interval == 5

    @e2e_pytest_unit
    def test_before_train_iter(self) -> None:
        """Test before_train_iter function."""

        hook = EvalBeforeRunHook()
        hook._original_interval = 4

        eval_hook = MockEvalHook()

        mock_runner = Config(
            {
                "hooks": [eval_hook],
            }
        )

        hook.before_train_iter(mock_runner)
        assert hook._original_interval is None
        assert eval_hook.interval == 4
