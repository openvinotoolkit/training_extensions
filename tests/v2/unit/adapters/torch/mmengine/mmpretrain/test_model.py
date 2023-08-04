# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from otx.v2.adapters.torch.mmengine.mmpretrain.model import configure_in_channels, get_model, list_models
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from pytest_mock.plugin import MockerFixture


@pytest.fixture()
def config() -> Config:
    return Config({
        "model": {
            "backbone": "ResNet",
            "neck": {
                "type": "FPN",
                "in_channels": -1,
            },
            "head": {
                "type": "ClsHead",
                "in_channels": -1,
            },
        },
    })

def test_configure_in_channels(config: Config, mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
    # Mock build_backbone and build_neck functions
    class MockBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shape = (1, 256, 14, 14)
            self.input_shapes = {"test1": (1, 256, 14, -1)}

        def forward(self, *args, **kwargs) -> list:
            _, _ = args, kwargs
            mock_tensor = mocker.MagicMock()
            mock_tensor.shape = (1, 256, 7, 7)
            return [[mock_tensor, mock_tensor], [mock_tensor, mock_tensor]]

    class MockNeck(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shape = (1, 128, 14, 14)

        def forward(self, *args, **kwargs) -> list:
            _, _ = args, kwargs
            mock_tensor = mocker.MagicMock()
            mock_tensor.shape = (1, 128, 7, 7)
            return [mock_tensor, mock_tensor]

    monkeypatch.setattr("otx.v2.adapters.torch.mmengine.mmpretrain.model.TRANSFORMER_BACKBONES", ["MockBackbone"])
    build_backbone_mock = mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.model.build_backbone")
    build_neck_mock = mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.model.build_neck")

    # Mock the output of the build_backbone function
    build_backbone_mock.return_value = MockBackbone()

    # Mock the output of the build_neck function
    build_neck_mock.return_value = MockNeck()

    # Call the function to be tested
    updated_config = configure_in_channels(config)

    # Check that the in_channels parameters have been updated
    assert updated_config["model"]["neck"]["in_channels"] == 256
    assert updated_config["model"]["head"]["in_channels"] == 128

    # Check that the build_backbone and build_neck functions were called with the correct arguments
    build_backbone_mock.assert_called_once_with("ResNet")
    build_neck_mock.assert_called_once_with({
        "type": "FPN",
        "in_channels": 256,
    })

    mock_config = Config({})
    updated_config = configure_in_channels(config=mock_config)
    assert updated_config == mock_config

    mock_config = Config({"head": {"in_channels": -1}})
    with pytest.raises(KeyError):
        configure_in_channels(config=mock_config)


def test_get_model(mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
    # Mock the configure_in_channels function
    configure_in_channels_mock = mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.model.configure_in_channels")
    configure_in_channels_mock.return_value = mocker.MagicMock()

    # Mock the get_mmpretrain_model function
    class MockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shape = (1, 256, 14, 14)

        def forward(self, *args, **kwargs) -> list:
            _, _ = args, kwargs
            return []

        def to(self, memory_format: torch.channels_last) -> torch.nn.Module:
            _ = memory_format
            return self
    get_mmpretrain_model_mock = mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.model.get_mmpretrain_model")
    get_mmpretrain_model_mock.return_value = MockModel()

    # Call the function to be tested
    get_model("resnet50", pretrained=True, num_classes=10, channel_last=True)

    # Check that the get_mmpretrain_model function was called with the correct arguments
    get_mmpretrain_model_mock.assert_called_once_with(
        "resnet50",
        pretrained=True,
    )

    model_dict = {
        "model": {
            "name": "test1",
            "backbone": "resnet50",
            "neck": {
                "type": "FPN",
                "in_channels": -1,
            },
            "head": {
                "type": "ClsHead",
                "in_channels": -1,
                "num_classes": 10,
            },
        },
    }
    model = get_model(model_dict, pretrained=True, num_classes=10, channel_last=True)
    configure_in_channels_mock.assert_called_once()
    assert isinstance(model, torch.nn.Module)

    model_config = Config(model_dict)
    get_model(model_config, pretrained=True, num_classes=10, channel_last=True)
    # Check that the configure_in_channels function was called with the correct arguments
    configure_in_channels_mock.assert_called()

    fromfile_mock = mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.model.Config.fromfile", return_value=model_config)
    monkeypatch.setattr("otx.v2.adapters.torch.mmengine.mmpretrain.model.MODEL_CONFIGS", {"test2": "test.yaml"})
    get_model("test2", pretrained=True, num_classes=10, channel_last=True)
    fromfile_mock.assert_called_once()

    mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.model.Path.is_file", return_value=True)
    model = get_model("model/template.yaml", pretrained=True, num_classes=10, channel_last=True)
    assert isinstance(model, torch.nn.Module)

    mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.model.configure_in_channels", return_value=Config(model_dict["model"]))
    model = get_model("model/template.yaml", pretrained=True, num_classes=10, channel_last=True)
    assert isinstance(model, torch.nn.Module)

def test_list_models(mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
    mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.model.list_mmpretrain_model", return_value=["mm_model1", "mm_model2"])
    monkeypatch.setattr("otx.v2.adapters.torch.mmengine.mmpretrain.model.MODEL_CONFIGS", {"otx_1": "test.yaml"})

    result = list_models()
    assert result == ["mm_model1", "mm_model2", "otx_1"]

    result = list_models(pattern="otx*")
    assert result == ["otx_1"]
