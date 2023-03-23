# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest

from otx.core.ov.ops.sorting_maximization import (
    NonMaxSuppressionV5,
    NonMaxSuppressionV9,
    TopKV3,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestTopKV3:
    @e2e_pytest_unit
    def test_invalid_attr(self):
        with pytest.raises(ValueError):
            TopKV3("dummy", shape=(1,), axis=1, mode="error", sort="value", index_element_type="i32")

        with pytest.raises(ValueError):
            TopKV3("dummy", shape=(1,), axis=1, mode="max", sort="error", index_element_type="i32")

        with pytest.raises(ValueError):
            TopKV3("dummy", shape=(1,), axis=1, mode="max", sort="value", index_element_type="f32")

    @e2e_pytest_unit
    def test_forward(self):
        op = TopKV3("dummy", shape=(1,), axis=1, mode="max", sort="value")
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy")


class TestNonMaxSuppressionV5:
    @e2e_pytest_unit
    def test_forward(self):
        op = NonMaxSuppressionV5("dummy", shape=(1,))
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy", "dummy")


class TestNonMaxSuppressionV9:
    @e2e_pytest_unit
    def test_forward(self):
        op = NonMaxSuppressionV9("dummy", shape=(1,))
        with pytest.raises(NotImplementedError):
            op("dummy", "dummy", "dummy")
