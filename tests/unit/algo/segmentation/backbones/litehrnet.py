from copy import deepcopy

import pytest
import torch
from otx.algo.segmentation.model.backbones.litehrnet import LiteHRNet, NeighbourSupport, SpatialWeightingV2, StemV2


class TestSpatialWeightingV2:
    def test_forward(self) -> None:
        swv2 = SpatialWeightingV2(channels=32)
        assert swv2 is not None

        inputs = torch.randn(1, 32, 32, 32)
        outputs = swv2(inputs)
        assert outputs is not None


class TestStemV2:
    @pytest.fixture()
    def stemv2(self) -> StemV2:
        return StemV2(in_channels=32, stem_channels=32, out_channels=32, expand_ratio=1)

    def test_init(self) -> None:
        stemv2_extra_stride = StemV2(
            in_channels=32, stem_channels=32, out_channels=32, expand_ratio=1, extra_stride=True,
        )
        assert stemv2_extra_stride is not None

        stemv2_input_norm = StemV2(in_channels=32, stem_channels=32, out_channels=32, expand_ratio=1, input_norm=True)
        assert stemv2_input_norm is not None

    def test_forward(self, stemv2) -> None:
        inputs = torch.randn(1, 32, 32, 32)
        outputs = stemv2(inputs)
        assert outputs is not None


class TestNeighbourSupport:
    def test_forward(self) -> None:
        neighbour_support = NeighbourSupport(channels=32)
        assert neighbour_support is not None

        inputs = torch.randn(1, 32, 32, 32)
        outputs = neighbour_support(inputs)
        assert outputs is not None


class TestLiteHRNet:
    @pytest.fixture()
    def extra_cfg(self) -> dict:
        return {
            "stem": {
                "stem_channels": 32,
                "out_channels": 32,
                "expand_ratio": 1,
                "strides": (2, 2),
                "extra_stride": False,
                "input_norm": False,
            },
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

    @pytest.fixture()
    def backbone(self, extra_cfg) -> LiteHRNet:
        return LiteHRNet(extra=extra_cfg)

    def test_init(self, extra_cfg)-> None:
        extra = deepcopy(extra_cfg)

        extra["add_stem_features"] = True
        model = LiteHRNet(extra=extra)
        assert model is not None

        extra["stages_spec"]["module_type"] = ("NAIVE", "NAIVE", "NAIVE")
        extra["stages_spec"]["weighting_module_version"] = "v2"
        model = LiteHRNet(extra=extra)
        assert model is not None

    def test_init_weights(self, backbone)-> None:
        backbone.init_weights()

        with pytest.raises(TypeError):
            backbone.init_weights(0)

    def test_forward(self, extra_cfg, backbone)-> None:
        backbone.train()
        inputs = torch.randn((1, 3, 224, 224))
        outputs = backbone(inputs)
        assert outputs is not None

        extra = deepcopy(extra_cfg)
        extra["stages_spec"]["module_type"] = ("NAIVE", "NAIVE", "NAIVE")
        extra["stages_spec"]["weighting_module_version"] = "v2"
        model = LiteHRNet(extra=extra)
        outputs = model(inputs)
        assert outputs is not None
