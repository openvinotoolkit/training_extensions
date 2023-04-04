# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest
import torch
from torch.nn import functional as F

from otx.core.ov.ops.convolutions import ConvolutionV1, GroupConvolutionV1
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestConvolutionV1:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)
        self.weight = torch.randn((128, 3, 3, 3))

    @e2e_pytest_unit
    def test_invalid_attr(self):
        with pytest.raises(ValueError):
            ConvolutionV1(
                "dummy",
                shape=self.shape,
                strides=[1, 1],
                pads_begin=[0, 0],
                pads_end=[0, 0],
                dilations=[1, 1],
                auto_pad="error",
            )

    @e2e_pytest_unit
    def test_forward(self):

        op = ConvolutionV1(
            "dummy",
            shape=self.shape,
            strides=[1, 1],
            pads_begin=[0, 0],
            pads_end=[0, 0],
            dilations=[1, 1],
            auto_pad="valid",
        )

        with pytest.raises(NotImplementedError):
            op(self.input, torch.randn((1, 1, 1, 1, 1, 1)))

        assert torch.equal(
            op(self.input, self.weight),
            F.conv2d(self.input, self.weight, None, op.attrs.strides, 0, op.attrs.dilations),
        )

        op = ConvolutionV1(
            "dummy",
            shape=self.shape,
            strides=[1, 1],
            pads_begin=[1, 1],
            pads_end=[1, 1],
            dilations=[1, 1],
            auto_pad="explicit",
        )
        assert torch.equal(
            op(self.input, self.weight),
            F.conv2d(self.input, self.weight, None, op.attrs.strides, 1, op.attrs.dilations),
        )


class TestGroupConvolutionV1:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)
        self.weight = torch.randn((3, 128, 1, 3, 3))

    @e2e_pytest_unit
    def test_invalid_attr(self):
        with pytest.raises(ValueError):
            GroupConvolutionV1(
                "dummy",
                shape=self.shape,
                strides=[1, 1],
                pads_begin=[0, 0],
                pads_end=[0, 0],
                dilations=[1, 1],
                auto_pad="error",
            )

    @e2e_pytest_unit
    def test_forward(self):

        op = GroupConvolutionV1(
            "dummy",
            shape=self.shape,
            strides=[1, 1],
            pads_begin=[0, 0],
            pads_end=[0, 0],
            dilations=[1, 1],
            auto_pad="valid",
        )

        with pytest.raises(NotImplementedError):
            op(self.input, torch.randn((1, 1, 1, 1, 1, 1, 1)))

        n_groups = self.weight.shape[0]
        weight = self.weight.view(-1, *self.weight.shape[2:])
        assert torch.equal(
            op(self.input, self.weight),
            F.conv2d(self.input, weight, None, op.attrs.strides, 0, op.attrs.dilations, n_groups),
        )

        op = GroupConvolutionV1(
            "dummy",
            shape=self.shape,
            strides=[1, 1],
            pads_begin=[1, 1],
            pads_end=[1, 1],
            dilations=[1, 1],
            auto_pad="explicit",
        )
        n_groups = self.weight.shape[0]
        weight = self.weight.view(-1, *self.weight.shape[2:])
        assert torch.equal(
            op(self.input, self.weight),
            F.conv2d(self.input, weight, None, op.attrs.strides, 1, op.attrs.dilations, n_groups),
        )
