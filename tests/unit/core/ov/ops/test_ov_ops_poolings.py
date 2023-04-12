# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest
import torch
from torch.nn import functional as F

from otx.core.ov.ops.poolings import AvgPoolV1, MaxPoolV0
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMaxPoolV0:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)

    @e2e_pytest_unit
    def test_invalid_attr(self):
        with pytest.raises(ValueError):
            MaxPoolV0(
                "dummy",
                shape=self.shape,
                strides=[1, 1],
                pads_begin=[0, 0],
                pads_end=[0, 0],
                kernel=[8, 8],
                rounding_type="error",
            )

        with pytest.raises(ValueError):
            MaxPoolV0(
                "dummy",
                shape=self.shape,
                strides=[1, 1],
                pads_begin=[0, 0],
                pads_end=[0, 0],
                kernel=[8, 8],
                auto_pad="error",
            )

        with pytest.raises(ValueError):
            MaxPoolV0(
                "dummy",
                shape=self.shape,
                strides=[1, 1],
                pads_begin=[0, 0],
                pads_end=[0, 0],
                kernel=[8, 8],
                index_element_type="error",
            )

        with pytest.raises(NotImplementedError):
            MaxPoolV0(
                "dummy",
                shape=self.shape,
                strides=[1, 1],
                pads_begin=[0, 0],
                pads_end=[0, 0],
                kernel=[8, 8],
                axis=1,
            )

    @e2e_pytest_unit
    def test_forward(self):
        op = MaxPoolV0(
            "dummy",
            shape=self.shape,
            strides=[1, 1],
            pads_begin=[0, 0],
            pads_end=[0, 0],
            kernel=[8, 8],
        )

        with pytest.raises(NotImplementedError):
            op(torch.randn(1, 1, 1, 1, 1, 1))

        output = op(self.input)
        ref = F.max_pool2d(
            self.input,
            op.attrs.kernel,
            op.attrs.strides,
            dilation=op.attrs.dilations,
            ceil_mode=op.attrs.rounding_type == "ceil",
            return_indices=True,
        )
        for i, j in zip(output, ref):
            assert torch.equal(i, j)


class TestAvgPoolV1:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)

    @e2e_pytest_unit
    def test_invalid_attr(self):
        with pytest.raises(ValueError):
            AvgPoolV1(
                "dummy",
                shape=self.shape,
                exclude_pad=False,
                strides=[1, 1],
                pads_begin=[0, 0],
                pads_end=[0, 0],
                kernel=[8, 8],
                rounding_type="error",
            )

        with pytest.raises(ValueError):
            AvgPoolV1(
                "dummy",
                shape=self.shape,
                exclude_pad=False,
                strides=[1, 1],
                pads_begin=[0, 0],
                pads_end=[0, 0],
                kernel=[8, 8],
                auto_pad="error",
            )

    @e2e_pytest_unit
    def test_forward(self):
        op = AvgPoolV1(
            "dummy",
            shape=self.shape,
            exclude_pad=False,
            strides=[1, 1],
            pads_begin=[0, 0],
            pads_end=[0, 0],
            kernel=[8, 8],
        )

        with pytest.raises(NotImplementedError):
            op(torch.randn(1, 1, 1, 1, 1, 1))

        output = op(self.input)
        ref = F.avg_pool2d(
            self.input,
            op.attrs.kernel,
            op.attrs.strides,
            ceil_mode=op.attrs.rounding_type == "ceil",
            count_include_pad=not op.attrs.exclude_pad,
        )
        assert torch.equal(output, ref)
