"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.checkpoint_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.utils import Config

from otx.algorithms.common.adapters.mmcv.hooks.checkpoint_hook import (
    CheckpointHookWithValResults,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockRunner:
    class _MockModel:
        def __init__(self, string):
            self.name = string

        def buffers(self):
            return None

    class _MockLogger:
        def info(self, string):
            print(string)

    def __init__(self):
        self.model = self._MockModel("model")
        self.ema_model = self._MockModel("ema_model")
        self.meta = {}
        self.epoch = 1
        self.iter = 1
        self.save_ckpt = True
        self.logger = self._MockLogger()
        self.save_ema_model = True

    def save_checkpoint(self, *args, **kwrags):
        pass


class TestCheckpointHookWithValResults:
    """Test class for CheckpointHookWithValResults."""

    @e2e_pytest_unit
    def test_before_run(self) -> None:
        """Test before_run function."""

        hook = CheckpointHookWithValResults()
        mock_runner = Config({"work_dir": "./temp_dir/"})
        hook.before_run(mock_runner)
        assert hook.out_dir == "./temp_dir/"

    @e2e_pytest_unit
    def test_after_train_epoch(self, mocker) -> None:
        """Test after_train_epoch function."""

        mocker.patch.object(CheckpointHookWithValResults, "every_n_epochs", return_value=True)
        mocker.patch("otx.algorithms.common.adapters.mmcv.hooks.checkpoint_hook.allreduce_params", return_value=True)
        hook = CheckpointHookWithValResults(sync_buffer=True, out_dir="./tmp_dir/")
        runner = MockRunner()
        hook.after_train_epoch(runner)

        assert runner.model.name == "model"
        assert runner.save_ema_model is False
