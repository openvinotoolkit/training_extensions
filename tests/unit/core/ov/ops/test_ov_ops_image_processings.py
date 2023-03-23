# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest
import torch
from torch.nn import functional as F

from otx.core.ov.ops.image_processings import InterpolateV4
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestInterpolateV4:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 3, 64, 64)
        self.input = torch.randn(self.shape)

    @e2e_pytest_unit
    def test_forward(self):

        with pytest.raises(ValueError):
            op = InterpolateV4(
                "dummy",
                shape=self.shape,
                mode="error",
                shape_calculation_mode="sizes",
            )

        with pytest.raises(ValueError):
            op = InterpolateV4(
                "dummy",
                shape=self.shape,
                mode="nearest",
                shape_calculation_mode="error",
            )

        with pytest.raises(ValueError):
            op = InterpolateV4(
                "dummy",
                shape=self.shape,
                mode="nearest",
                shape_calculation_mode="sizes",
                coordinate_transformation_mode="error",
            )

        with pytest.raises(ValueError):
            op = InterpolateV4(
                "dummy",
                shape=self.shape,
                mode="nearest",
                shape_calculation_mode="sizes",
                nearest_mode="error",
            )

        op = InterpolateV4(
            "dummy",
            shape=self.shape,
            mode="nearest",
            shape_calculation_mode="sizes",
        )
        assert torch.equal(
            op(self.input, torch.tensor((4, 3, 128, 128)), torch.tensor((1, 1, 2, 2))),
            F.interpolate(self.input, (128, 128), mode="nearest"),
        )

        op = InterpolateV4(
            "dummy",
            shape=self.shape,
            mode="nearest",
            shape_calculation_mode="scales",
        )
        assert torch.equal(
            op(self.input, torch.tensor((4, 3, 128, 128)), torch.tensor((1, 1, 2, 2))),
            F.interpolate(self.input, scale_factor=(2, 2), mode="nearest"),
        )

        op = InterpolateV4(
            "dummy",
            shape=self.shape,
            mode="linear",
            shape_calculation_mode="sizes",
        )
        assert torch.equal(
            op(self.input, torch.tensor((4, 3, 128, 128)), torch.tensor((1, 1, 2, 2))),
            F.interpolate(self.input, (128, 128), mode="bilinear", align_corners=False),
        )

        op = InterpolateV4(
            "dummy",
            shape=self.shape,
            mode="cubic",
            shape_calculation_mode="sizes",
        )
        assert torch.equal(
            op(self.input, torch.tensor((4, 3, 128, 128)), torch.tensor((1, 1, 2, 2))),
            F.interpolate(self.input, (128, 128), mode="bicubic", align_corners=False),
        )
