from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from otx.algo.segmentation.backbones import LiteHRNetBackbone
from otx.algo.segmentation.backbones.litehrnet import NeighbourSupport, NNLiteHRNet, SpatialWeightingV2, StemV2


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
            in_channels=32,
            stem_channels=32,
            out_channels=32,
            expand_ratio=1,
            extra_stride=True,
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


class TestNNLiteHRNet:
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
            "out_modules": {
                "conv": {
                    "enable": True,
                    "channels": 320,
                },
                "position_att": {
                    "enable": True,
                    "key_channels": 128,
                    "value_channels": 320,
                    "psp_size": [1, 3, 6, 8],
                },
                "local_att": {
                    "enable": False,
                },
            },
        }

    @pytest.fixture()
    def backbone(self, extra_cfg) -> NNLiteHRNet:
        return NNLiteHRNet(**extra_cfg)

    def test_init(self, extra_cfg) -> None:
        extra = deepcopy(extra_cfg)

        extra["add_stem_features"] = True
        model = NNLiteHRNet(**extra)
        assert model is not None

        extra["stages_spec"]["module_type"] = ("NAIVE", "NAIVE", "NAIVE")
        extra["stages_spec"]["weighting_module_version"] = "v2"
        model = NNLiteHRNet(extra=extra)
        assert model is not None

    def test_init_weights(self, backbone) -> None:
        backbone.init_weights()

        with pytest.raises(TypeError):
            backbone.init_weights(0)

    def test_forward(self, extra_cfg, backbone) -> None:
        backbone.train()
        inputs = torch.randn((1, 3, 224, 224))
        outputs = backbone(inputs)
        assert outputs is not None

        extra = deepcopy(extra_cfg)
        extra["stages_spec"]["module_type"] = ("NAIVE", "NAIVE", "NAIVE")
        extra["stages_spec"]["weighting_module_version"] = "v2"
        model = NNLiteHRNet(extra=extra)
        outputs = model(inputs)
        assert outputs is not None

    @pytest.fixture()
    def mock_load_from_http(self, mocker) -> MagicMock:
        return mocker.patch.object(LiteHRNetBackbone, "load_from_http")

    @pytest.fixture()
    def mock_load_checkpoint_to_model(self, mocker) -> MagicMock:
        return mocker.patch.object(LiteHRNetBackbone, "load_checkpoint_to_model")

    @pytest.fixture()
    def pretrained_weight(self, tmp_path) -> str:
        weight = tmp_path / "pretrained.pth"
        weight.touch()
        return str(weight)

    @pytest.fixture()
    def mock_torch_load(self, mocker) -> MagicMock:
        return mocker.patch("otx.algo.segmentation.backbones.mscan.torch.load")

    def test_load_pretrained_weights(
        self,
        extra_cfg,
        pretrained_weight,
        mock_torch_load,
        mock_load_checkpoint_to_model,
    ):
        extra_cfg["add_stem_features"] = True
        model = NNLiteHRNet(extra=extra_cfg)
        model.load_pretrained_weights(pretrained=pretrained_weight)

        mock_torch_load.assert_called_once_with(pretrained_weight, "cpu")
        mock_load_checkpoint_to_model.assert_called_once()

    def test_load_pretrained_weights_from_url(self, extra_cfg, mock_load_from_http, mock_load_checkpoint_to_model):
        pretrained_weight = "www.fake.com/fake.pth"
        extra_cfg["add_stem_features"] = True
        model = NNLiteHRNet(extra=extra_cfg)
        model.load_pretrained_weights(pretrained=pretrained_weight)

        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        mock_load_from_http.assert_called_once_with(filename=pretrained_weight, map_location="cpu", model_dir=cache_dir)
        mock_load_checkpoint_to_model.assert_called_once()
