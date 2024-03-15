"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.model_ema_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math

import torch
from mmcv.runner import BaseRunner
from mmcv.runner.hooks.ema import EMAHook

from otx.algorithms.common.adapters.mmcv.hooks import (
    CustomModelEMAHook,
    DualModelEMAHook,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3),
            torch.nn.Conv2d(8, 16, 3),
            torch.nn.Conv2d(16, 64, 3),
        )

    def forward(self, x):
        return self.model(x)


class MockRunner(BaseRunner):
    class _DualModel:
        model_s = MockModel()
        model_t = MockModel()

    def __init__(self):
        self.model = self._DualModel()
        self._epoch = 0
        self._iter = 0
        self.data_loader = range(1000)

    def run(self):
        pass

    def save_checkpoint(self):
        pass

    def train(self):
        pass

    def val(self):
        pass


class TestDualModelEMAHook:
    """Test class for DualModelEMAHook."""

    @e2e_pytest_unit
    def test_before_run(self) -> None:
        """Test before_run function."""

        hook = DualModelEMAHook()
        runner = MockRunner()
        hook.before_run(runner)

    @e2e_pytest_unit
    def test_before_train_epoch(self) -> None:
        """Test before_train_epoch function."""

        hook = DualModelEMAHook(epoch_momentum=0.99)
        hook.enabled = True
        runner = MockRunner()
        hook.before_train_epoch(runner)
        assert hook.momentum == 1 - math.pow((1 - 0.99), (1 / 1000))
        assert hook.epoch_momentum == 0.0

    @e2e_pytest_unit
    def test_after_train_iter(self) -> None:
        """Test after_train_iter function."""

        hook = DualModelEMAHook()
        runner = MockRunner()
        # Not enable
        hook.after_train_iter(runner)

        hook.enabled = True
        hook.interval = 5

        runner._iter = 9
        # Skip
        hook.after_train_iter(runner)

        runner._iter = 0
        hook.before_run(runner)
        runner._iter = 10
        # Just copy
        hook.after_train_iter(runner)

        runner._iter = 0
        hook.before_run(runner)
        runner._epoch = 9
        # EMA
        hook.after_train_iter(runner)

    @e2e_pytest_unit
    def test_after_train_epoch(self):
        """Test after_train_epoch function."""

        hook = DualModelEMAHook()
        runner = MockRunner()
        hook.before_run(runner)
        hook.after_train_epoch(runner)


class TestCustomModelEMAHook:
    @e2e_pytest_unit
    def test_before_train_epoch(self, mocker):
        """Test before_train_epoch function."""

        mocker.patch.object(EMAHook, "before_train_epoch", return_value=True)
        hook = CustomModelEMAHook(epoch_momentum=0.99)
        runner = MockRunner()
        hook.before_train_epoch(runner)
        assert hook.momentum == 1 - math.pow((1 - 0.99), (1 / 1000))
        assert hook.epoch_momentum == 0.0
