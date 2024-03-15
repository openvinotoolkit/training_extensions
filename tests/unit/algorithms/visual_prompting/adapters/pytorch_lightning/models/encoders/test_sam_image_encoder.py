"""Tests sam image encoder used for visual prompting task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from tests.test_suite.e2e_test_system import e2e_pytest_unit
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.encoders.sam_image_encoder import SAMImageEncoder
import pytest
from omegaconf import DictConfig
import torch.nn as nn
import torch


class MockBackbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backbone = nn.Linear(1, 1)

    def forward(self, *args, **kwargs):
        return torch.Tensor([[1]])


class TestSAMImageEncoder:
    @pytest.fixture()
    def config(self, mocker) -> DictConfig:
        return DictConfig(dict(image_size=1024))

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "backbone,expected",
        [
            ("tiny_vit", "TinyViT"),
            ("vit_b", "ViT"),
        ],
    )
    def test_new(self, config: DictConfig, backbone: str, expected: str) -> None:
        """Test __new__."""
        config.update({"backbone": backbone})

        sam_image_encoder = SAMImageEncoder(config)

        assert sam_image_encoder.__class__.__name__ == expected
