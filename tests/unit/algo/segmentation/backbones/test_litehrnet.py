from unittest.mock import MagicMock

import pytest
import torch
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
    def cfg(self) -> dict:
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
    def backbone(self, cfg) -> NNLiteHRNet:
        return NNLiteHRNet(**cfg)

    @pytest.fixture()
    def mock_torch_load(self, mocker) -> MagicMock:
        return mocker.patch("otx.algo.segmentation.backbones.litehrnet.torch.load")

    @pytest.fixture()
    def mock_load_from_http(self, mocker) -> MagicMock:
        return mocker.patch("otx.algo.segmentation.backbones.litehrnet.load_from_http")

    @pytest.fixture()
    def mock_load_checkpoint_to_model(self, mocker) -> MagicMock:
        return mocker.patch("otx.algo.segmentation.backbones.litehrnet.load_checkpoint_to_model")

    @pytest.fixture()
    def pretrained_weight(self, tmp_path) -> str:
        weight = tmp_path / "pretrained.pth"
        weight.touch()
        return str(weight)

    def test_init(self, cfg) -> None:
        model = NNLiteHRNet(**cfg)
        assert model is not None

    def test_forward(self, cfg, backbone) -> None:
        backbone.train()
        inputs = torch.randn((1, 3, 224, 224))
        outputs = backbone(inputs)
        assert outputs is not None

    def test_load_pretrained_weights_from_url(
        self,
        mock_load_from_http,
        mock_load_checkpoint_to_model,
        backbone,
    ) -> None:
        pretrained_weight = "www.fake.com/fake.pth"
        backbone.load_pretrained_weights(pretrained=pretrained_weight)
        mock_load_from_http.assert_called_once()
        mock_load_checkpoint_to_model.assert_called_once()

    def test_load_pretrained_weights(
        self,
        cfg,
        pretrained_weight,
        mock_torch_load,
        mock_load_checkpoint_to_model,
    ):
        model = NNLiteHRNet(**cfg)
        model.load_pretrained_weights(pretrained=pretrained_weight)

        mock_torch_load.assert_called_once_with(pretrained_weight, "cpu")
        mock_load_checkpoint_to_model.assert_called_once()
