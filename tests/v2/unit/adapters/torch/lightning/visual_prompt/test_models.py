# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import OrderedDict

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import DictConfig, OmegaConf
from otx.v2.adapters.torch.lightning.visual_prompt.model import get_model, list_models
from otx.v2.adapters.torch.lightning.visual_prompt.registry import VisualPromptRegistry
from pytest_mock.plugin import MockerFixture

SAM_TINY_MODEL = {
    "model": {
        "name": "SAM_tiny",
        "image_size": 1024,
        "mask_threshold": 0.,
        "return_logits": True,
        "backbone": "tiny_vit",
        "loss_type": "sam",
    },
}

SAM_MODEL = {
    "name": "SAM",
    "image_size": 1024,
    "mask_threshold": 0.,
    "return_logits": True,
    "backbone": "vit",
    "loss_type": "sam",
}


class MockModel:
    def __init__(self, config: DictConfig, state_dict: OrderedDict) -> None:
        self.config = config
        self.state_dict = state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MockModel2:
    def __init__(self, config: DictConfig, state_dict: OrderedDict) -> None:
        self.config = config
        self.state_dict = state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

registry = VisualPromptRegistry()
registry.register_module(name="SAM_tiny", module=MockModel)
registry.register_module(name="SAM", module=MockModel2)

def test_get_model(mocker: MockerFixture, monkeypatch: MonkeyPatch, tmp_dir_path: Path) -> None:
    sam_tiny_config_file = tmp_dir_path / "test_sam_tiny.yaml"
    OmegaConf.save(config=DictConfig(SAM_TINY_MODEL), f=sam_tiny_config_file)

    monkeypatch.setattr(
        "otx.v2.adapters.torch.lightning.visual_prompt.model.MODEL_CONFIGS",
        {
            "SAM_tiny": str(sam_tiny_config_file),
        },
    )
    monkeypatch.setattr(
        "otx.v2.adapters.torch.lightning.visual_prompt.model.MODELS",
        registry,
    )

    # from name
    result = get_model(model="SAM_tiny")
    assert result.__class__.__name__ == "MockModel"

    # from path
    result = get_model(model=str(sam_tiny_config_file))
    assert result.__class__.__name__ == "MockModel"

    # from dict with model key
    result = get_model(model=SAM_TINY_MODEL)
    assert result.__class__.__name__ == "MockModel"

    # from dict without model key
    result = get_model(model=SAM_MODEL)
    assert result.__class__.__name__ == "MockModel2"

    # from DictConfig
    result = get_model(model=DictConfig(SAM_TINY_MODEL))
    assert result.__class__.__name__ == "MockModel"

    # from checkpoint
    mock_load = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.model.torch.load", return_value={"model": {"state_dict": {}}})
    result = get_model(model="SAM_tiny", checkpoint="checkpoint.pth")
    assert result.__class__.__name__ == "MockModel"
    mock_load.assert_called_once_with("checkpoint.pth")

    # invalid model - SAM_2
    mock_config = {
        "name": "SAM_2",
        "image_size": 1024,
        "mask_threshold": 0.,
        "return_logits": True,
        "backbone": "vit",
        "loss_type": "sam",
    }
    with pytest.raises(NotImplementedError, match="is not implemented."):
        get_model(model=mock_config)

def test_list_models(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "otx.v2.adapters.torch.lightning.visual_prompt.model.MODEL_CONFIGS",
        {
            "otx_sam_tiny": "test.yaml",
            "otx_model_1": "test_2.yaml",
            "model_1": "test_2.yaml",
        },
    )

    result = list_models()
    assert result.sort() == ["otx_sam_tiny", "otx_model_1", "model_1"].sort()

    result = list_models(pattern="otx*")
    assert result.sort() == ["otx_sam_tiny", "otx_model_1"].sort()
