"""Test cross focal loss."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from otx.algorithms.detection.adapters.mmdet.models.losses.cross_focal_loss import (
    CrossSigmoidFocalLoss,
)


class TestCrossFocalLoss:
    """Test cross focal loss."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create the loss object"""
        self.predictions = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.labels = torch.tensor([1, 1, 2])
        self.loss = CrossSigmoidFocalLoss(num_classes=3)

    def test_loss_computation(self):
        """Test loss output."""
        assert np.round(self.loss(self.predictions, self.labels).item(), decimals=4) == 0.0885

    @pytest.mark.xfail(reason="This should fail as the masks are different")
    def test_loss_computation_with_mask(self):
        """Tests loss output with ignored label is provided."""
        valid_label_mask1 = torch.ones((3, 3))
        loss1 = self.loss(self.predictions, self.labels, valid_label_mask=valid_label_mask1)
        valid_label_mask2 = torch.zeros((3, 3))
        loss2 = self.loss(self.predictions, self.labels, valid_label_mask=valid_label_mask2)
        assert loss1 != loss2

    def test_reduction(self):
        """Test reduction."""

        loss1 = self.loss(self.predictions, self.labels, reduction_override="none")
        loss2 = self.loss(self.predictions, self.labels, reduction_override="mean")
        loss3 = self.loss(self.predictions, self.labels, reduction_override="sum")
        assert loss1.shape == (3, 3)
        assert loss2 != loss3
