"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.eval_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from mmcv.runner import BaseRunner
from torch.utils.data import DataLoader

from otx.algorithms.common.adapters.mmcv.hooks.eval_hook import (
    CustomEvalHook,
    DistCustomEvalHook,
    single_gpu_test,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockDataloader(DataLoader):
    """Mock class for pytorch dataloader."""

    class _MockDataset:
        def __init__(self, data) -> None:
            self.data = data

        def __len__(self) -> None:
            return len(self.data)

        def evaluate(self, results, *args, **kwargs):
            if len(results) == 0:
                return {"top-1": 0.5}
            return {"top-1": 0.7}

    def __init__(self) -> None:
        self.data = [{"img": torch.randn(1, 3, 224, 224)}]
        self.dataset = self._MockDataset(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)


class MockRunner(BaseRunner):
    """Mock class for BaseRunner."""

    class _ModuleWrapper:
        module = torch.nn.Module()

    class _MockBuffer:
        output = {}
        ready = False

        def clear(self):
            self.output = {}

    def __init__(self):
        self.model = torch.nn.Module()
        self.ema_model = self._ModuleWrapper()
        self._epoch = 9
        self._iter = 9
        self.log_buffer = self._MockBuffer()
        self.logger = None
        self.work_dir = "./tmp_dir/"

    def run(self):
        pass

    def save_checkpoint(self):
        pass

    def train(self):
        pass

    def val(self):
        pass


class TestCustomEvalHook:
    """Test class for CustomEvalHook."""

    @e2e_pytest_unit
    def test_init(self) -> None:
        """Test __init__ function."""

        hook = CustomEvalHook(metric="accuracy", dataloader=MockDataloader())
        assert hook.metric == "top-1"
        hook = CustomEvalHook(metric=["class_accuracy"], dataloader=MockDataloader())
        assert hook.metric == "accuracy"
        hook = CustomEvalHook(metric=["accuracy"], dataloader=MockDataloader())
        assert hook.metric == "top-1"

    @e2e_pytest_unit
    def test_do_evaluate(self, mocker) -> None:
        """Test do_evaluate function."""

        hook = CustomEvalHook(metric="accuracy", dataloader=MockDataloader())
        runner = MockRunner()
        mocker.patch("otx.algorithms.common.adapters.mmcv.hooks.eval_hook.single_gpu_test", return_value=[])
        mocker.patch.object(CustomEvalHook, "evaluate", return_value=True)
        hook._do_evaluate(runner, ema=False)
        hook.ema_eval_start_epoch = 3
        hook._do_evaluate(runner, ema=True)

    @e2e_pytest_unit
    def test_after_train_epoch(self, mocker) -> None:
        """Test after_train_epoch function."""

        hook = CustomEvalHook(metric="accuracy", dataloader=MockDataloader())
        runner = MockRunner()
        hook.by_epoch = True
        hook.interval = 4
        mocker.patch.object(CustomEvalHook, "_do_evaluate", return_value=True)
        hook.after_train_epoch(runner)
        hook.interval = 5
        hook.after_train_epoch(runner)

    @e2e_pytest_unit
    def test_after_train_iter(self, mocker) -> None:
        """Test after_train_iter function."""

        hook = CustomEvalHook(metric="accuracy", dataloader=MockDataloader())
        runner = MockRunner()
        hook.by_epoch = False
        hook.interval = 4
        mocker.patch.object(CustomEvalHook, "_do_evaluate", return_value=True)
        hook.after_train_iter(runner)
        hook.interval = 5
        hook.after_train_iter(runner)

    @e2e_pytest_unit
    def test_evaluate(self) -> None:
        """Test evaluate function."""

        hook = CustomEvalHook(metric="accuracy", dataloader=MockDataloader())
        runner = MockRunner()
        hook.evaluate(runner, results=[], results_ema=None)
        assert runner.log_buffer.output["top-1"] == 0.5
        assert runner.log_buffer.ready is True
        assert hook.best_score == 0.5
        assert runner.save_ckpt is True

        hook.evaluate(runner, results=[], results_ema=[0])
        assert runner.log_buffer.output["top-1_EMA"] == 0.7
        assert runner.save_ema_model is True


@e2e_pytest_unit
def test_single_gpu_test() -> None:
    """Test function for single_gpu_test."""

    class _MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args, **kwargs):
            return torch.Tensor([0])

    model = _MockModel()
    single_gpu_test(model, MockDataloader())


class TestDistCustomEvalHook:
    """Test class for DistCustomEvalHook."""

    @e2e_pytest_unit
    def test_do_evaluate(self, mocker) -> None:
        """Test _do_evaluate function."""

        mocker.patch("mmcls.apis.multi_gpu_test", return_value=True)
        mocker.patch.object(DistCustomEvalHook, "evaluate", return_value=True)
        runner = MockRunner()
        dataloader = MockDataloader()
        runner._rank = 0
        hook = DistCustomEvalHook(dataloader, metric="top-1")
        hook._do_evaluate(runner)

        with pytest.raises(TypeError):
            hook = DistCustomEvalHook(None)

    @e2e_pytest_unit
    def test_after_train_epoch(self, mocker) -> None:
        """Test after_train_epoch function."""

        dataloader = MockDataloader()
        hook = DistCustomEvalHook(dataloader, metric="top-1")
        mocker.patch.object(DistCustomEvalHook, "_do_evaluate", side_effect=RuntimeError("VALID ERROR"))
        hook.by_epoch = True
        hook.interval = 5
        runner = MockRunner()
        with pytest.raises(RuntimeError):
            hook.after_train_epoch(runner)

        hook.interval = 3
        hook.after_train_epoch(runner)

    @e2e_pytest_unit
    def test_after_train_iter(self, mocker) -> None:
        """Test after_train_iter function."""

        dataloader = MockDataloader()
        hook = DistCustomEvalHook(dataloader, metric="top-1")
        mocker.patch.object(DistCustomEvalHook, "_do_evaluate", side_effect=RuntimeError("VALID ERROR"))
        hook.by_epoch = False
        hook.interval = 5
        runner = MockRunner()
        with pytest.raises(RuntimeError):
            hook.after_train_iter(runner)

        hook.interval = 3
        hook.after_train_iter(runner)
