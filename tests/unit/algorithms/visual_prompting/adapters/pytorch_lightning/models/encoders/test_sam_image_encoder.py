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
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        self.mocker_backbone = mocker.patch(
            "otx.algorithms.visual_prompting.adapters.pytorch_lightning.models.encoders.sam_image_encoder.build_vit",
            return_value=MockBackbone(),
        )

        self.base_config = DictConfig(dict(backbone="vit_b", image_size=1024))

    @e2e_pytest_unit
    @pytest.mark.parametrize("backbone", ["vit_b", "resnet"])
    def test_init(self, backbone: str):
        """Test init."""
        self.mocker_backbone.reset_mock()

        config = self.base_config.copy()
        config.update(dict(backbone=backbone))

        if backbone == "resnet":
            with pytest.raises(NotImplementedError):
                sam_image_encoder = SAMImageEncoder(config)
        else:
            sam_image_encoder = SAMImageEncoder(config)
            self.mocker_backbone.assert_called_once()

    @e2e_pytest_unit
    def test_forward(self, mocker):
        """Test forward."""
        self.mocker_backbone.reset_mock()

        sam_image_encoder = SAMImageEncoder(self.base_config)
        mocker_forward = mocker.patch.object(sam_image_encoder.backbone, "forward")
        sam_image_encoder.forward(torch.Tensor([1.0]))

        mocker_forward.assert_called_once()
