# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of ScaleNorm."""

import torch
from otx.algo.keypoint_detection.utils.scale_norm import ScaleNorm


class TestScaleNorm:
    def test_forward(self) -> None:
        """Test forward."""
        scale_norm = ScaleNorm(dim=64)

        assert scale_norm.scale == 0.125
        assert torch.all(scale_norm.g == torch.ones(1))

        inputs = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)

        result = scale_norm(inputs)
        assert result.shape == inputs.shape
