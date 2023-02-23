"""Unit test for otx.mpa.modules.hooks.recording_forward_hooks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
import torch

from otx.mpa.modules.hooks.recording_forward_hooks import (
    ActivationMapHook,
    BaseRecordingForwardHook,
    DetSaliencyMapHook,
    EigenCamHook,
    FeatureVectorHook,
    ReciproCAMHook,
)
from otx.mpa.modules.models.heads.custom_atss_head import CustomATSSHead
from otx.mpa.modules.models.heads.custom_ssd_head import CustomSSDHead
from otx.mpa.modules.models.heads.custom_vfnet_head import CustomVFNetHead
from otx.mpa.modules.models.heads.custom_yolox_head import CustomYOLOXHead
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


class TestDetSaliencyMapHook:
    """Test class for DetSaliencyMapHook."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        class _MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.with_neck = True
                self.neck = torch.nn.Module()
                self.neck.forward = self.forward
                self.bbox_head = torch.nn.Module()
                self.bbox_head.cls_out_channels = 3

            def forward(self, x):
                return x

        self.module = _MockModule()
        self.hook = DetSaliencyMapHook(self.module)

    @e2e_pytest_unit
    def test_func(self, mocker) -> None:
        """Test func function."""

        mocker.patch.object(
            DetSaliencyMapHook, "_get_cls_scores_from_feature_map", return_value=[torch.randn(1, 3, 14, 14)]
        )
        assert self.hook.func(torch.randn(1, 3, 14, 14)) is not None

    @e2e_pytest_unit
    def test_get_cls_scores_from_feature_map(self) -> None:
        """Test _get_cls_scores_from_feature_map function."""

        self.module.bbox_head = CustomATSSHead(num_classes=3, in_channels=64)
        self.hook = DetSaliencyMapHook(self.module)
        assert self.hook._get_cls_scores_from_feature_map(torch.Tensor(1, 3, 64, 32, 32)) is not None
        self.module.bbox_head = CustomYOLOXHead(num_classes=3, in_channels=64)
        self.hook = DetSaliencyMapHook(self.module)
        assert self.hook._get_cls_scores_from_feature_map(torch.Tensor(1, 3, 64, 32, 32)) is not None
        self.module.bbox_head = CustomVFNetHead(num_classes=3, in_channels=64)
        self.module.bbox_head.anchor_generator.num_base_anchors = 1
        self.hook = DetSaliencyMapHook(self.module)
        assert self.hook._get_cls_scores_from_feature_map(torch.Tensor(1, 3, 64, 32, 32)) is not None
        self.module.bbox_head = CustomSSDHead(
            anchor_generator=dict(
                type="SSDAnchorGenerator",
                basesize_ratio_range=(0.15, 0.9),
                strides=(16, 32, 48),
                ratios=[[0.5], [0.1], [0.3]],
            ),
            act_cfg={},
        )
        self.hook = DetSaliencyMapHook(self.module)
        assert self.hook._get_cls_scores_from_feature_map(torch.Tensor(1, 3, 512, 32, 32)) is not None
        self.module.bbox_head = torch.nn.Module()
        self.module.bbox_head.cls_out_channels = 3
        self.hook = DetSaliencyMapHook(self.module)
        with pytest.raises(NotImplementedError):
            self.hook._get_cls_scores_from_feature_map(torch.Tensor(1, 3, 512, 32, 32))


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
