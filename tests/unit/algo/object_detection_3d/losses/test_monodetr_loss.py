# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit test for MonoDETR loss."""
import torch
from otx.algo.object_detection_3d.losses.monodetr_loss import MonoDETRCriterion


class TestMonoDETRCriterion:
    def test_loss_labels(self):
        criterion = MonoDETRCriterion(num_classes=10, weight_dict={}, focal_alpha=0.5)
        outputs = {
            "scores": torch.randn(2, 10, 10),
        }
        targets = [
            {"labels": torch.tensor([1, 2, 0, 0, 0, 0, 1, 2, 0, 0])},
            {"labels": torch.tensor([3, 4, 0, 0, 0, 0, 3, 4, 0, 0])},
        ]
        indices = [
            (torch.tensor([0, 1]), torch.tensor([0, 1])),
            (torch.tensor([0, 1]), torch.tensor([0, 1])),
        ]
        num_boxes = 4

        loss = criterion.loss_labels(outputs, targets, indices, num_boxes)
        assert "loss_ce" in loss
        assert isinstance(loss["loss_ce"], torch.Tensor)

    def test_loss_3dcenter(self):
        criterion = MonoDETRCriterion(num_classes=10, weight_dict={}, focal_alpha=0.5)
        outputs = {
            "boxes_3d": torch.randn(2, 10, 4),
        }
        targets = [
            {"boxes_3d": torch.tensor([[1, 2], [3, 4]])},
            {"boxes_3d": torch.tensor([[5, 6], [7, 8]])},
        ]
        indices = [
            (torch.tensor([0, 1]), torch.tensor([0, 1])),
            (torch.tensor([0, 1]), torch.tensor([0, 1])),
        ]
        num_boxes = 4

        loss = criterion.loss_3dcenter(outputs, targets, indices, num_boxes)
        assert "loss_center" in loss
        assert isinstance(loss["loss_center"], torch.Tensor)

    def test_forward(self):
        criterion = MonoDETRCriterion(num_classes=10, weight_dict={}, focal_alpha=0.5)
        outputs = {
            "scores": torch.randn(1, 100, 10),
            "boxes_3d": torch.randn(1, 100, 6),
            "depth": torch.randn(1, 100, 2),
            "size_3d": torch.randn(1, 100, 3),
            "heading_angle": torch.randn(1, 100, 24),
            "pred_depth_map_logits": torch.randn(1, 100, 80, 80),
        }
        targets = [
            {
                "labels": torch.tensor([0, 0, 0, 0]),
                "boxes": torch.tensor(
                    [
                        [0.7697, 0.4923, 0.0398, 0.0663],
                        [0.7371, 0.4857, 0.0339, 0.0620],
                        [0.7126, 0.4850, 0.0246, 0.0501],
                        [0.5077, 0.5280, 0.0444, 0.1475],
                    ],
                ),
                "depth": torch.tensor([[47.5800], [55.2600], [62.3900], [23.7700]]),
                "size_3d": torch.tensor(
                    [
                        [1.5500, 1.3700, 3.9700],
                        [1.6900, 1.7400, 3.7600],
                        [1.5500, 1.3900, 3.5500],
                        [1.6200, 1.6300, 4.5000],
                    ],
                ),
                "heading_angle": torch.tensor(
                    [
                        [2.0000e00, 4.6737e-02],
                        [8.0000e00, 1.2180e-01],
                        [8.0000e00, 1.5801e-01],
                        [9.0000e00, 1.8260e-04],
                    ],
                ),
                "boxes_3d": torch.tensor(
                    [
                        [0.7689, 0.4918, 0.0191, 0.0208, 0.0327, 0.0336],
                        [0.7365, 0.4858, 0.0163, 0.0175, 0.0310, 0.0310],
                        [0.7122, 0.4848, 0.0118, 0.0127, 0.0248, 0.0252],
                        [0.5089, 0.5234, 0.0235, 0.0209, 0.0693, 0.0783],
                    ],
                ),
            },
        ]

        losses = criterion.forward(outputs, targets)
        assert isinstance(losses, dict)
        assert len(losses) == 8
        for loss in losses.values():
            assert isinstance(loss, torch.Tensor)
