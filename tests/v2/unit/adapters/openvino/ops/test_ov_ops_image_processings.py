# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.v2.adapters.openvino.ops.image_processings import InterpolateV4
from torch.nn import functional


class TestInterpolateV4:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 3, 64, 64)
        self.input = torch.randn(self.shape)


    def test_forward(self) -> None:

        with pytest.raises(ValueError, match="Invalid mode error."):
            op = InterpolateV4(
                "dummy",
                shape=self.shape,
                mode="error",
                shape_calculation_mode="sizes",
            )

        with pytest.raises(ValueError, match="Invalid shape_calculation_mode error."):
            op = InterpolateV4(
                "dummy",
                shape=self.shape,
                mode="nearest",
                shape_calculation_mode="error",
            )

        with pytest.raises(ValueError, match="Invalid coordinate_transformation_mode error."):
            op = InterpolateV4(
                "dummy",
                shape=self.shape,
                mode="nearest",
                shape_calculation_mode="sizes",
                coordinate_transformation_mode="error",
            )

        with pytest.raises(ValueError, match="Invalid nearest_mode error."):
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
            functional.interpolate(self.input, (128, 128), mode="nearest"),
        )

        op = InterpolateV4(
            "dummy",
            shape=self.shape,
            mode="nearest",
            shape_calculation_mode="scales",
        )
        assert torch.equal(
            op(self.input, torch.tensor((4, 3, 128, 128)), torch.tensor((1, 1, 2, 2))),
            functional.interpolate(self.input, scale_factor=(2, 2), mode="nearest"),
        )

        op = InterpolateV4(
            "dummy",
            shape=self.shape,
            mode="linear",
            shape_calculation_mode="sizes",
        )
        assert torch.equal(
            op(self.input, torch.tensor((4, 3, 128, 128)), torch.tensor((1, 1, 2, 2))),
            functional.interpolate(self.input, (128, 128), mode="bilinear", align_corners=False),
        )

        op = InterpolateV4(
            "dummy",
            shape=self.shape,
            mode="cubic",
            shape_calculation_mode="sizes",
        )
        assert torch.equal(
            op(self.input, torch.tensor((4, 3, 128, 128)), torch.tensor((1, 1, 2, 2))),
            functional.interpolate(self.input, (128, 128), mode="bicubic", align_corners=False),
        )
