"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.cancel_interface_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import EpochBasedRunner

from otx.algorithms.common.adapters.mmcv.hooks.cancel_hook import CancelInterfaceHook
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def mock_callback(*args):
    pass


class TestCancelInterfaceHook:
    """Test class for CancelInterfaceHook"""

    @e2e_pytest_unit
    def test_cancel(self):
        """Test cancel function."""

        class MockRunner(EpochBasedRunner):
            def __init__(self):
                self._max_epoch = 10
                self.should_stop = False
                self._epoch = 5

        hook = CancelInterfaceHook(mock_callback)
        assert hook.cancel() is None

        mock_runner = MockRunner()
        mock_runner.should_stop = True
        hook.runner = mock_runner
        assert hook.cancel() is None

        mock_runner.should_stop = False
        hook.runner = mock_runner
        hook.cancel()
        assert hook.runner.should_stop is True
        assert hook.runner._max_epochs == hook.runner.epoch

    @e2e_pytest_unit
    def test_before_run(self) -> None:
        hook = CancelInterfaceHook(mock_callback)
        runner = "RUNNER"

        hook.before_run(runner)
        assert hook.runner == "RUNNER"

    @e2e_pytest_unit
    def test_after_run(self) -> None:
        hook = CancelInterfaceHook(mock_callback)
        hook.runner = "RUNNER"
        hook.after_run("RUNNER")
        assert hook.runner is None
