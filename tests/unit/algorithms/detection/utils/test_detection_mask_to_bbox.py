# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.common.utils.mask_to_bbox import mask2bbox, mask_to_border
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMask2Bbox:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mask = np.zeros((3, 3))
        self.mask[0, 0] = 1

    @e2e_pytest_unit
    def test_mask_to_border(self):
        out = mask_to_border(self.mask)
        assert (out == self.mask).all()

    @e2e_pytest_unit
    def test_mask2bbox(self):
        out = mask2bbox(self.mask)
        expected_out = [[0, 0, 1, 1]]
        assert out == expected_out
