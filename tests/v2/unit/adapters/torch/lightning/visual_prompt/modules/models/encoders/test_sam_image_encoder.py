"""Tests sam image encoder used for visual prompting task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig
from otx.v2.adapters.torch.lightning.visual_prompt.modules.models.encoders.sam_image_encoder import SAMImageEncoder
from torch import nn


class MockBackbone(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        _, _ = args, kwargs
        self.backbone = nn.Linear(1, 1)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        _, _ = args, kwargs
        return torch.Tensor([[1]])


class TestSAMImageEncoder:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        self.mocker_backbone = mocker.patch(
            "otx.v2.adapters.torch.lightning.visual_prompt.modules.models.encoders.sam_image_encoder.build_vit",
            return_value=MockBackbone(),
        )

        self.base_config = DictConfig({"backbone": "vit_b", "image_size": 1024})

    @pytest.mark.parametrize("backbone", ["vit_b", "resnet"])
    def test_init(self, backbone: str) -> None:
        """Test init."""
        self.mocker_backbone.reset_mock()

        config = self.base_config.copy()
        config.update({"backbone": backbone})

        if backbone == "resnet":
            with pytest.raises(NotImplementedError):
                SAMImageEncoder(config)
        else:
            SAMImageEncoder(config)
            self.mocker_backbone.assert_called_once()

    def test_forward(self, mocker) -> None:
        """Test forward."""
        self.mocker_backbone.reset_mock()

        sam_image_encoder = SAMImageEncoder(self.base_config)
        mocker_forward = mocker.patch.object(sam_image_encoder.backbone, "forward")
        sam_image_encoder.forward(torch.Tensor([1.0]))

        mocker_forward.assert_called_once()
