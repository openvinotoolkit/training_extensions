# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from otx.v2.adapters.openvino.ops.sorting_maximization import (
    NonMaxSuppressionV5,
    NonMaxSuppressionV9,
    TopKV3,
)


class TestTopKV3:

    def test_invalid_attr(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode error."):
            TopKV3("dummy", shape=(1,), axis=1, mode="error", sort="value", index_element_type="i32")

        with pytest.raises(ValueError, match="Invalid sort error."):
            TopKV3("dummy", shape=(1,), axis=1, mode="max", sort="error", index_element_type="i32")

        with pytest.raises(ValueError, match="Invalid index_element_type"):
            TopKV3("dummy", shape=(1,), axis=1, mode="max", sort="value", index_element_type="f32")


    def test_forward(self) -> None:
        op = TopKV3("dummy", shape=(1,), axis=1, mode="max", sort="value")
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy")


class TestNonMaxSuppressionV5:

    def test_forward(self) -> None:
        op = NonMaxSuppressionV5("dummy", shape=(1,))
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy", "dummy")


class TestNonMaxSuppressionV9:

    def test_forward(self) -> None:
        op = NonMaxSuppressionV9("dummy", shape=(1,))
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy", "dummy")
