# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.v2.adapters.openvino.ops.poolings import AvgPoolV1, MaxPoolV0
from torch.nn import functional


class TestMaxPoolV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)


    def test_invalid_attr(self) -> None:
        with pytest.raises(ValueError, match="Invalid rounding_type error."):
            MaxPoolV0(
                "dummy",
                shape=self.shape,
                strides=[1, 1],
                pads_begin=[0, 0],
                pads_end=[0, 0],
                kernel=[8, 8],
                rounding_type="error",
            )

        with pytest.raises(ValueError, match="Invalid auto_pad error."):
            MaxPoolV0(
                "dummy",
                shape=self.shape,
                strides=[1, 1],
                pads_begin=[0, 0],
                pads_end=[0, 0],
                kernel=[8, 8],
                auto_pad="error",
            )

        with pytest.raises(ValueError, match="Invalid index_element_type error."):
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


    def test_forward(self) -> None:
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
        ref = functional.max_pool2d(
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
    def setup(self) -> None:
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)


    def test_invalid_attr(self) -> None:
        with pytest.raises(ValueError, match="Invalid rounding_type error."):
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

        with pytest.raises(ValueError, match="Invalid auto_pad error."):
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


    def test_forward(self) -> None:
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
        ref = functional.avg_pool2d(
            self.input,
            op.attrs.kernel,
            op.attrs.strides,
            ceil_mode=op.attrs.rounding_type == "ceil",
            count_include_pad=not op.attrs.exclude_pad,
        )
        assert torch.equal(output, ref)
