# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from mmpretrain.registry import MODELS
from omegaconf import DictConfig
from otx.core.utils.build import (
    build_mm_model,
    get_classification_layers,
    get_default_num_async_infer_requests,
    modify_num_classes,
)


@pytest.fixture()
def fxt_mm_config() -> DictConfig:
    return DictConfig(
        {
            "backbone": {
                "version": "b0",
                "pretrained": True,
                "type": "OTXEfficientNet",
            },
            "head": {
                "in_channels": 1280,
                "loss": {
                    "loss_weight": 1.0,
                    "type": "CrossEntropyLoss",
                },
                "num_classes": 1000,
                "topk": (1, 5),
                "type": "LinearClsHead",
            },
            "neck": {
                "type": "GlobalAveragePooling",
            },
            "data_preprocessor": {
                "mean": [123.678, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
                "to_rgb": False,
                "type": "ClsDataPreprocessor",
            },
            "type": "ImageClassifier",
        },
    )


def test_build_mm_model(fxt_mm_config, mocker) -> None:
    model = build_mm_model(config=fxt_mm_config, model_registry=MODELS)
    assert model.__class__.__name__ == "ImageClassifier"

    mock_load_checkpoint = mocker.patch("mmengine.runner.load_checkpoint")
    model = build_mm_model(config=fxt_mm_config, model_registry=MODELS, load_from="path/to/weights.pth")
    mock_load_checkpoint.assert_called_once_with(model, "path/to/weights.pth", map_location="cpu")


def test_get_default_num_async_infer_requests() -> None:
    # Test the get_default_num_async_infer_requests function.

    # Mock os.cpu_count() to return a specific value
    original_cpu_count = os.cpu_count
    os.cpu_count = lambda: 4

    # Call the function and check the return value
    assert get_default_num_async_infer_requests() == 2

    # Restore the original os.cpu_count() function
    os.cpu_count = original_cpu_count

    # Check the warning message
    with pytest.warns(UserWarning, match="Set the default number of OpenVINO inference requests"):
        get_default_num_async_infer_requests()


def test_get_classification_layers(fxt_mm_config) -> None:
    expected_result = {
        "head.fc.weight": {"stride": 1, "num_extra_classes": 0},
        "head.fc.bias": {"stride": 1, "num_extra_classes": 0},
    }

    result = get_classification_layers(fxt_mm_config, MODELS)
    assert result == expected_result


def test_modify_num_classes():
    config = DictConfig({"num_classes": 10, "model": {"num_classes": 5}})
    num_classes = 7
    modify_num_classes(config, num_classes)
    assert config["num_classes"] == num_classes
    assert config["model"]["num_classes"] == num_classes

    config = DictConfig({"num_classes": 10, "model": {"num_classes": 5}})
    num_classes = 7
    modify_num_classes(config, num_classes)
    assert config["num_classes"] == num_classes
    assert config["model"]["num_classes"] == num_classes

    config = DictConfig({"model": {"layers": [{"units": 64}]}})
    num_classes = 7
    modify_num_classes(config, num_classes)
    assert "num_classes" not in config
    assert config["model"]["layers"][0]["units"] == 64
