# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from getitune.backend.lightning.models.segmentation.backbones.litehrnet import (
    LiteHRNetModule,
    NeighbourSupport,
    SpatialWeightingV2,
    Stem,
)


class TestSpatialWeightingV2:
    def test_forward(self) -> None:
        swv2 = SpatialWeightingV2(channels=32)
        assert swv2 is not None

        inputs = torch.randn(1, 32, 32, 32)
        outputs = swv2(inputs)
        assert outputs is not None


class TestStem:
    @pytest.fixture
    def stem(self) -> Stem:
        return Stem(in_channels=32, stem_channels=32, out_channels=32, expand_ratio=1)

    def test_init(self) -> None:
        stem_extra_stride = Stem(
            in_channels=32,
            stem_channels=32,
            out_channels=32,
            expand_ratio=1,
            extra_stride=True,
        )
        assert stem_extra_stride is not None

        stem_input_norm = Stem(in_channels=32, stem_channels=32, out_channels=32, expand_ratio=1, input_norm=True)
        assert stem_input_norm is not None

    def test_forward(self, stem) -> None:
        inputs = torch.randn(1, 32, 32, 32)
        outputs = stem(inputs)
        assert outputs is not None


class TestNeighbourSupport:
    def test_forward(self) -> None:
        neighbour_support = NeighbourSupport(channels=32)
        assert neighbour_support is not None

        inputs = torch.randn(1, 32, 32, 32)
        outputs = neighbour_support(inputs)
        assert outputs is not None


class TestLiteHRNetModule:
    @pytest.fixture
    def cfg(self) -> dict:
        return {
            "stem_configuration": {},
            "num_stages": 3,
            "stages_spec": {
                "num_modules": (2, 4, 2),
                "num_branches": (2, 3, 4),
                "num_blocks": (2, 2, 2),
                "module_type": ("LITE", "LITE", "LITE"),
                "with_fuse": (True, True, True),
                "reduce_ratios": (8, 8, 8),
                "num_channels": [
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                ],
            },
        }

    @pytest.fixture
    def backbone(self, cfg) -> LiteHRNetModule:
        return LiteHRNetModule(**cfg)

    def test_init(self, cfg) -> None:
        model = LiteHRNetModule(**cfg)
        assert model is not None

    def test_forward(self, cfg, backbone) -> None:
        backbone.train()
        inputs = torch.randn((1, 3, 224, 224))
        outputs = backbone(inputs)
        assert outputs is not None
