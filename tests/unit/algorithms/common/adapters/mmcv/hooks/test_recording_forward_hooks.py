"""Unit test for otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
import torch

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ActivationMapHook,
    BaseRecordingForwardHook,
    EigenCamHook,
    FeatureVectorHook,
    ReciproCAMHook,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockBaseRecordingForwardHook(BaseRecordingForwardHook):
    @staticmethod
    def func(*args):
        return torch.Tensor([[0]])


class TestBaseRecordingForwardHook:
    """Test class for BaseRecordingForwardHook"""

    @e2e_pytest_unit
    def test_records(self, mocker) -> None:
        """Test records function."""

        hook = MockBaseRecordingForwardHook(torch.nn.Module())
        assert hook.records == []

    @e2e_pytest_unit
    def test_func(self) -> None:
        """Test func function."""
        hook = MockBaseRecordingForwardHook(torch.nn.Module())
        assert hook.func() == torch.Tensor([[0]])

    @e2e_pytest_unit
    def test_recording_forward(self) -> None:
        """Test _recording_forward."""

        hook = MockBaseRecordingForwardHook(torch.nn.Module())
        hook._recording_forward(torch.nn.Module(), torch.Tensor([0]), torch.Tensor([0]))
        assert hook._records == [np.array([0.0])]

    @e2e_pytest_unit
    def test_enter(self) -> None:
        """Test __enter__ function."""

        class _MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.Module()

        hook = MockBaseRecordingForwardHook(_MockModule())
        hook.__enter__()

    @e2e_pytest_unit
    def test_exit(self) -> None:
        """Test __exit__ function."""

        class MockHandle:
            def remove(self):
                pass

        hook = MockBaseRecordingForwardHook(torch.nn.Module())
        hook._handle = MockHandle()
        hook.__exit__(None, None, None)


class TestEigenCamHook:
    """Test class for EigenCamHook."""

    def test_func(self) -> None:
        """Test func function."""

        hook = EigenCamHook(torch.nn.Module())
        feature_map = torch.randn(8, 3, 14, 14)
        assert hook.func(feature_map) is not None


class TestActivationMapHook:
    """Test class for ActivationMapHook."""

    def test_func(self) -> None:
        """Test func function."""

        hook = ActivationMapHook(torch.nn.Module())
        feature_map = torch.randn(8, 3, 14, 14)
        assert hook.func(feature_map) is not None


class TestFeatureVectorHook:
    """Test class for FeatureVectorHook."""

    def test_func(self) -> None:
        """Test func function."""

        hook = FeatureVectorHook(torch.nn.Module())
        feature_map = torch.randn(8, 3, 14, 14)
        assert hook.func(feature_map) is not None


class TestReciproCAMHook:
    """Test class for ReciproCAMHook."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        class _MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.with_neck = True
                self.neck = torch.nn.Module()
                self.neck.forward = self.forward
                self.head = torch.nn.Module()
                self.head.num_classes = 3
                self.head.simple_test = self.forward

            def forward(self, x):
                return x

        self.module = _MockModule()
        self.hook = ReciproCAMHook(self.module)

    @e2e_pytest_unit
    def test_func(self) -> None:
        """Test func function."""

        assert self.hook.func([torch.randn(1, 3, 1, 1)]) is not None
