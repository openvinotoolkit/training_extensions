# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of DetrCriterion."""

import pytest
import torch
from otx.algo.detection.losses.rtdetr_loss import DetrCriterion


class TestDetrCriterion:
    @pytest.fixture()
    def criterion(self):
        weight_dict = {"loss_vfl": 1.0, "loss_bbox": 5, "loss_giou": 2}
        return DetrCriterion(weight_dict, num_classes=2)

    @pytest.fixture()
    def outputs(self):
        return {
            "pred_boxes": torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]]),
            "pred_logits": torch.tensor([[[0.9, 0.1], [0.2, 0.8]]]),
        }

    @pytest.fixture()
    def targets(self):
        return [
            {"boxes": torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]), "labels": torch.tensor([1, 0])},
        ]

    def test_loss_labels_vfl(self, criterion, outputs, targets):
        indices = [(torch.tensor([0]), torch.tensor([1]))]
        num_boxes = 2

        loss_dict = criterion.loss_labels_vfl(outputs, targets, indices, num_boxes)

        assert "loss_vfl" in loss_dict
        assert isinstance(loss_dict["loss_vfl"], torch.Tensor)

    def test_loss_boxes(self, criterion, outputs, targets):
        indices = [(torch.tensor([0]), torch.tensor([1]))]
        num_boxes = 2

        loss_dict = criterion.loss_boxes(outputs, targets, indices, num_boxes)

        assert "loss_bbox" in loss_dict
        assert "loss_giou" in loss_dict
        assert isinstance(loss_dict["loss_bbox"], torch.Tensor)
        assert isinstance(loss_dict["loss_giou"], torch.Tensor)

    def test_forward(self, criterion, outputs, targets):
        loss_dict = criterion.forward(outputs, targets)

        assert isinstance(loss_dict, dict)
        assert "loss_vfl" in loss_dict
        assert "loss_bbox" in loss_dict
        assert "loss_giou" in loss_dict
        assert isinstance(loss_dict["loss_vfl"], torch.Tensor)
        assert isinstance(loss_dict["loss_bbox"], torch.Tensor)
        assert isinstance(loss_dict["loss_giou"], torch.Tensor)
