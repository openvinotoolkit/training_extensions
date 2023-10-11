# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from otx.v2.adapters.torch.modules.utils.mask_to_bbox import mask2bbox, mask_to_border


class TestMask2Bbox:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.mask = np.zeros((3, 3))
        self.mask[0, 0] = 1

    def test_mask_to_border(self) -> None:
        out = mask_to_border(self.mask)
        assert (out == self.mask).all()

    def test_mask2bbox(self) -> None:
        out = mask2bbox(self.mask)
        expected_out = [[0, 0, 1, 1]]
        assert out == expected_out
