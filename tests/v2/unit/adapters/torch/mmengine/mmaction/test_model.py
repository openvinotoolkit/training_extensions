"""Unit-test for the model API for MMAction."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from otx.v2.adapters.torch.mmengine.mmaction.model import get_model, list_models
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config


@pytest.fixture()
def config() -> Config:
    return Config({
        "model": {
            "type": "Recognizer3D",
            "backbone": {
                "type": "X3D",
            },
            "cls_head": {
                "type": "X3DHead",
                "in_channels": 432,
                "num_classes": 400,
            },
        },
    })


def test_get_model(config) -> None:
    model_dict = {
        "model": {
            "type": "Recognizer3D",
            "backbone": {
                "type": "X3D",
            },
            "cls_head": {
                "type": "X3DHead",
                "in_channels": 432,
                "num_classes": 400,
            },
        },
    }
    model = get_model(model_dict, pretrained=True, num_classes=5)
    assert isinstance(model, torch.nn.Module)
    assert model.cls_head.num_classes == 5
    
    model = get_model(config, pretrained=True, num_classes=10)
    assert isinstance(model, torch.nn.Module)
    assert model.cls_head.num_classes == 10

def test_list_models(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("otx.v2.adapters.torch.mmengine.mmaction.model.MODEL_CONFIGS", {"otx_1": "test.yaml", "1_otx": "test2.yaml"})
    result = list_models()
    assert result == ["1_otx", "otx_1"]
    
    result = list_models(pattern="otx*")
    assert result == ["otx_1"]