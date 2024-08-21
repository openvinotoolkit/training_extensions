from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from otx.algo.segmentation.backbones import dinov2 as target_file
from otx.algo.segmentation.backbones.dinov2 import DinoVisionTransformer


class TestDinoVisionTransformer:
    @pytest.fixture()
    def mock_backbone_named_parameters(self) -> dict[str, MagicMock]:
        named_parameter = {}
        for i in range(3):
            parameter = MagicMock()
            parameter.requires_grad = True
            named_parameter[f"layer_{i}"] = parameter
        return named_parameter

    @pytest.fixture()
    def mock_backbone(self, mock_backbone_named_parameters) -> MagicMock:
        backbone = MagicMock()
        backbone.named_parameters.return_value = list(mock_backbone_named_parameters.items())
        return backbone

    @pytest.fixture(autouse=True)
    def mock_torch_hub_load(self, mocker, mock_backbone):
        return mocker.patch("otx.algo.segmentation.backbones.dinov2.torch.hub.load", return_value=mock_backbone)

    def test_init(self, mock_backbone, mock_backbone_named_parameters):
        dino = DinoVisionTransformer(name="dinov2_vits14", freeze_backbone=True, out_index=[8, 9, 10, 11])

        assert dino.backbone == mock_backbone
        for parameter in mock_backbone_named_parameters.values():
            assert parameter.requires_grad is False

    @pytest.fixture()
    def dino_vit(self) -> DinoVisionTransformer:
        return DinoVisionTransformer(
            name="dinov2_vits14",
            freeze_backbone=True,
            out_index=[8, 9, 10, 11],
        )

    def test_forward(self, dino_vit, mock_backbone):
        tensor = torch.rand(10, 3, 3, 3)
        dino_vit.forward(tensor)

        mock_backbone.assert_called_once_with(tensor)

    @pytest.fixture()
    def mock_load_from_http(self, mocker) -> MagicMock:
        return mocker.patch.object(target_file, "load_from_http")

    @pytest.fixture()
    def mock_load_checkpoint_to_model(self, mocker) -> MagicMock:
        return mocker.patch.object(target_file, "load_checkpoint_to_model")

    @pytest.fixture()
    def pretrained_weight(self, tmp_path) -> str:
        weight = tmp_path / "pretrained.pth"
        weight.touch()
        return str(weight)

    @pytest.fixture()
    def mock_torch_load(self, mocker) -> MagicMock:
        return mocker.patch("otx.algo.segmentation.backbones.mscan.torch.load")

    def test_load_pretrained_weights(self, dino_vit, pretrained_weight, mock_torch_load, mock_load_checkpoint_to_model):
        dino_vit.load_pretrained_weights(pretrained=pretrained_weight)
        mock_torch_load.assert_called_once_with(pretrained_weight, "cpu")
        mock_load_checkpoint_to_model.assert_called_once()

    def test_load_pretrained_weights_from_url(self, dino_vit, mock_load_from_http, mock_load_checkpoint_to_model):
        pretrained_weight = "www.fake.com/fake.pth"
        dino_vit.load_pretrained_weights(pretrained=pretrained_weight)

        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        mock_load_from_http.assert_called_once_with(filename=pretrained_weight, map_location="cpu", model_dir=cache_dir)
        mock_load_checkpoint_to_model.assert_called_once()
