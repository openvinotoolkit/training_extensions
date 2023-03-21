from copy import deepcopy

import pytest
import torch

from otx.algorithms.common.adapters.mmcv.configs.backbones.lite_hrnet_18 import (
    model as model_cfg,
)
from otx.algorithms.segmentation.adapters.mmseg.models.backbones.litehrnet import (
    LiteHRNet,
    NeighbourSupport,
    SpatialWeightingV2,
    StemV2,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestSpatialWeightingV2:
    @e2e_pytest_unit
    def test_forward(self):
        swv2 = SpatialWeightingV2(channels=32)
        assert swv2 is not None

        inputs = torch.randn(1, 32, 32, 32)
        outputs = swv2(inputs)
        assert outputs is not None


class TestStemV2:
    def setup_method(self):
        self.stemv2 = StemV2(in_channels=32, stem_channels=32, out_channels=32, expand_ratio=1)

    @e2e_pytest_unit
    def test_init(self):
        stemv2_extra_stride = StemV2(
            in_channels=32, stem_channels=32, out_channels=32, expand_ratio=1, extra_stride=True
        )
        assert stemv2_extra_stride is not None

        stemv2_input_norm = StemV2(in_channels=32, stem_channels=32, out_channels=32, expand_ratio=1, input_norm=True)
        assert stemv2_input_norm is not None

    @e2e_pytest_unit
    def test_forward(self):
        inputs = torch.randn(1, 32, 32, 32)
        outputs = self.stemv2(inputs)
        assert outputs is not None


class TestNeighbourSupport:
    @e2e_pytest_unit
    def test_forward(self):
        neighbour_support = NeighbourSupport(channels=32)
        assert neighbour_support is not None

        inputs = torch.randn(1, 32, 32, 32)
        outputs = neighbour_support(inputs)
        assert outputs is not None


class TestLiteHRNet:
    def setup_method(self):
        self.extra = model_cfg["backbone"]["extra"]
        self.model = LiteHRNet(extra=self.extra)

    @e2e_pytest_unit
    def test_init(self):
        extra = deepcopy(self.extra)
        extra["out_modules"]["conv"]["enable"] = True
        model = LiteHRNet(extra=extra)
        assert model is not None

        extra["add_stem_features"] = True
        model = LiteHRNet(extra=extra)
        assert model is not None

        extra["stages_spec"]["module_type"] = ("NAIVE", "NAIVE", "NAIVE")
        extra["stages_spec"]["weighting_module_version"] = "v2"
        model = LiteHRNet(extra=extra)
        assert model is not None

    @e2e_pytest_unit
    def test_init_weights(self):
        self.model.init_weights()

        with pytest.raises(TypeError, match="pretrained must be a str or None"):
            self.model.init_weights(0)

    @e2e_pytest_unit
    def test_forward(self):
        self.model.train()
        inputs = torch.randn((1, 3, 224, 224))
        outputs = self.model(inputs)
        assert outputs is not None

        extra = deepcopy(self.extra)
        extra["stages_spec"]["module_type"] = ("NAIVE", "NAIVE", "NAIVE")
        extra["stages_spec"]["weighting_module_version"] = "v2"
        model = LiteHRNet(extra=extra)
        outputs = model(inputs)
        assert outputs is not None
