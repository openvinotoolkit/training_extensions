"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.no_bias_decay_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.utils import Config

from otx.algorithms.common.adapters.mmcv.hooks.no_bias_decay_hook import NoBiasDecayHook
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockRunner:
    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3),
            torch.nn.BatchNorm2d(8),
        )
        self.optimizer = Config(
            {
                "param_groups": [
                    {
                        "params": None,
                        "weight_decay": None,
                        "lr": 0.01,
                    }
                ]
            }
        )


class TestNoBiasDecayHook:
    @e2e_pytest_unit
    def test_before_train_epoch(self):
        """Test before_train_epoch function."""

        hook = NoBiasDecayHook()
        runner = MockRunner()
        hook.before_train_epoch(runner)
        assert len(runner.optimizer.param_groups) == 3
        assert runner.optimizer.param_groups[0]["params"] is not None
        assert runner.optimizer.param_groups[1]["params"] is not None
        assert runner.optimizer.param_groups[1]["weight_decay"] == 0.0
        assert runner.optimizer.param_groups[1]["lr"] == 0.02
        assert runner.optimizer.param_groups[2]["params"] is not None
        assert runner.optimizer.param_groups[2]["weight_decay"] == 0.0

    @e2e_pytest_unit
    def test_after_train_epoch(self):
        hook = NoBiasDecayHook()
        runner = MockRunner()
        hook.after_train_epoch(runner)
        assert runner.optimizer.param_groups[0]["params"] is not None
