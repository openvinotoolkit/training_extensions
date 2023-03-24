"""Test utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from otx.algorithms.segmentation.adapters.mmseg.models.utils import LossEqualizer


class TestLossEqualizer:
    """Test loss equalizer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create the loss object"""
        weights = {"loss_a": 1.0, "loss_b": 2.0, "loss_c": 3.0}
        self.loss_equalizer = LossEqualizer(weights, momentum=0)

    def test_loss_equalizer(self):
        """Test value"""
        losses = {
            "loss_a": torch.tensor(10.0),
            "loss_b": torch.tensor(4.0),
            "loss_c": torch.tensor(1.0),
        }
        result = {val.item() for val in self.loss_equalizer.reweight(losses).values()}
        assert result == {2.5, 5.0, 7.5}
