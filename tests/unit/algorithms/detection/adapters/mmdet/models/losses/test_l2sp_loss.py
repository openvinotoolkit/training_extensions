"""Test L2SP loss with ignore."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from otx.algorithms.detection.adapters.mmdet.models.losses.l2sp_loss import L2SPLoss


class DummyModel(nn.Module):
    """Creates a dummy model for testing purposes."""

    def __init__(self, set_init_weights: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(4, 1)
        self.cnn = nn.Conv2d(1, 1, 1)
        self._set_weights(set_init_weights)

    def _set_weights(self, set_init_weights: float = 0.0):
        """Sets weights of all the parameters with a given value."""
        for param in self.parameters():
            param.data.fill_(set_init_weights)


class TestL2SPLoss:
    """Test L2SP loss."""

    @pytest.fixture(scope="class")
    def model_checkpoint(self):
        """Create a model checkpoint."""
        with TemporaryDirectory() as tmp_path:
            model = DummyModel(set_init_weights=1.0)
            model_ckpt = tmp_path + "/model.ckpt"
            torch.save(model.state_dict(), model_ckpt)
            yield model_ckpt

    def test_loss(self, model_checkpoint):
        """Test loss output."""
        new_model = DummyModel(set_init_weights=0.0)
        loss = L2SPLoss(new_model, model_checkpoint)

        assert np.round(loss().item(), 4).item() == 0.0007

        new_model = DummyModel(set_init_weights=1.0)
        loss = L2SPLoss(new_model, model_checkpoint)
        assert loss().item() == 0.0

    def test_value_error(self, model_checkpoint):
        """Test ValueError."""
        model = DummyModel()

        with pytest.raises(ValueError):
            L2SPLoss(model, model_ckpt=None)

        with pytest.raises(ValueError):
            L2SPLoss(model, model_checkpoint, loss_weight=-1.0)

    def test_state_dict(self):
        """Test model load with weights stored under state_dict key."""
        with TemporaryDirectory() as tmp_path:
            model = DummyModel(set_init_weights=1.0)
            model_ckpt = tmp_path + "/model.ckpt"
            torch.save({"state_dict": model.state_dict()}, model_ckpt)
            new_model = DummyModel(set_init_weights=1.0)
            loss = L2SPLoss(new_model, model_ckpt)
            assert loss().item() == 0.0

    def test_backbone(self):
        """Test model load with backbone weights.

        This tests the case when the new model has backbone param whereas the source model as the backbone layers
        directly.
        """
        with TemporaryDirectory() as tmp_path:
            model = DummyModel()
            for param, value in resnet18(pretrained=True).named_children():
                setattr(model, param, value)
            model_ckpt = tmp_path + "/model.ckpt"
            torch.save(model.state_dict(), model_ckpt)

            new_model = DummyModel()
            setattr(new_model, "backbone", resnet18(pretrained=True))
            # new_model.load_state_dict(torch.load(model_ckpt))
            loss = L2SPLoss(new_model, model_ckpt)
            assert loss().item() == 0.0
