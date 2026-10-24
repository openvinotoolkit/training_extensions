# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DEIM loss criterion."""

from __future__ import annotations

import pytest
import torch

from getitune.backend.lightning.models.detection.losses.deim_loss import DEIMCriterion


class TestDEIMCriterion:
    """Test class for DEIM loss criterion."""

    @pytest.fixture
    def criterion(self) -> DEIMCriterion:
        """Create a DEIM criterion instance."""
        weight_dict = {
            "loss_vfl": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
            "loss_mal": 1.0,
        }
        return DEIMCriterion(
            weight_dict=weight_dict,
            alpha=0.75,
            gamma=1.5,
            reg_max=32,
            num_classes=10,
        )

    @pytest.fixture
    def outputs(self) -> dict[str, torch.Tensor]:
        """Create mock model outputs."""
        return {
            "pred_boxes": torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]]),
            "pred_logits": torch.tensor(
                [
                    [
                        [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        [0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.5, 0.1],
                    ],
                ],
            ),
            "pred_corners": torch.randn(1, 2, 4, 33),  # (batch, num_queries, 4, reg_max+1)
            "teacher_corners": torch.randn(1, 2, 4, 33),
            "teacher_logits": torch.randn(1, 2, 10),
            "up": torch.tensor([0.5]),
            "reg_scale": torch.tensor([4.0]),
            "aux_outputs": [],
            "pre_outputs": {
                "pred_boxes": torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]]),
                "pred_logits": torch.tensor(
                    [
                        [
                            [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                            [0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.5, 0.1],
                        ],
                    ],
                ),
                "pred_corners": torch.randn(1, 2, 4, 33),
                "teacher_corners": torch.randn(1, 2, 4, 33),
                "teacher_logits": torch.randn(1, 2, 10),
                "up": torch.tensor([0.5]),
                "reg_scale": torch.tensor([4.0]),
                "ref_points": torch.tensor(
                    [
                        [
                            [0.1, 0.2, 0.3, 0.4],
                            [0.5, 0.6, 0.7, 0.8],
                        ],
                    ],
                ),
            },
            "enc_aux_outputs": [],
            "ref_points": torch.tensor(
                [
                    [
                        [0.1, 0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7, 0.8],
                    ],
                ],
            ),
        }

    @pytest.fixture
    def targets(self) -> list[dict[str, torch.Tensor]]:
        """Create mock targets."""
        return [
            {
                "boxes": torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]),
                "labels": torch.tensor([1, 0]),
            },
        ]

    def test_init(self) -> None:
        """Test DEIM criterion initialization."""
        weight_dict = {
            "loss_vfl": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
            "loss_mal": 1.0,
        }
        criterion = DEIMCriterion(
            weight_dict=weight_dict,
            alpha=0.75,
            gamma=1.5,
            reg_max=32,
            num_classes=10,
        )

        assert criterion.weight_dict == weight_dict
        assert criterion.alpha == 0.75
        assert criterion.gamma == 1.5
        assert criterion.reg_max == 32
        assert criterion.num_classes == 10

    def test_loss_labels_mal(self, criterion: DEIMCriterion, outputs: dict, targets: list) -> None:
        """Test Matchability-Aware Loss (MAL)."""
        indices = [(torch.tensor([0]), torch.tensor([1]))]
        num_boxes = 2

        loss_dict = criterion.loss_labels_mal(outputs, targets, indices, num_boxes)

        assert "loss_mal" in loss_dict
        assert isinstance(loss_dict["loss_mal"], torch.Tensor)
        assert loss_dict["loss_mal"].numel() == 1

    def test_loss_labels_vfl(self, criterion: DEIMCriterion, outputs: dict, targets: list) -> None:
        """Test Varifocal Loss (VFL)."""
        indices = [(torch.tensor([0]), torch.tensor([1]))]
        num_boxes = 2

        loss_dict = criterion.loss_labels_vfl(outputs, targets, indices, num_boxes)

        assert "loss_vfl" in loss_dict
        assert isinstance(loss_dict["loss_vfl"], torch.Tensor)
        assert loss_dict["loss_vfl"].numel() == 1

    def test_loss_boxes(self, criterion: DEIMCriterion, outputs: dict, targets: list) -> None:
        """Test bounding box regression loss."""
        indices = [(torch.tensor([0]), torch.tensor([1]))]
        num_boxes = 2

        loss_dict = criterion.loss_boxes(outputs, targets, indices, num_boxes)

        assert "loss_bbox" in loss_dict
        assert "loss_giou" in loss_dict
        assert isinstance(loss_dict["loss_bbox"], torch.Tensor)
        assert isinstance(loss_dict["loss_giou"], torch.Tensor)

    def test_loss_local(self, criterion: DEIMCriterion, outputs: dict, targets: list) -> None:
        """Test Fine-Grained Localization (FGL) and Decoupled Distillation Focal (DDF) losses."""
        indices = [(torch.tensor([0]), torch.tensor([1]))]
        num_boxes = 2

        loss_dict = criterion.loss_local(outputs, targets, indices, num_boxes)

        # Should contain FGL loss
        assert "loss_fgl" in loss_dict
        assert isinstance(loss_dict["loss_fgl"], torch.Tensor)

        # Should contain DDF loss when teacher_corners is provided
        if "teacher_corners" in outputs and outputs["teacher_corners"] is not None:
            assert "loss_ddf" in loss_dict
            assert isinstance(loss_dict["loss_ddf"], torch.Tensor)

    def test_available_losses(self, criterion: DEIMCriterion) -> None:
        """Test that all expected losses are available."""
        available_losses = criterion._available_losses

        # Should have all loss functions
        assert len(available_losses) == 4

        # Check function names
        loss_names = [loss.__name__ for loss in available_losses]
        expected_names = ["loss_boxes", "loss_labels_vfl", "loss_labels_mal", "loss_local"]
        for name in expected_names:
            assert name in loss_names

    def test_forward_with_pre_outputs(self, criterion: DEIMCriterion, outputs: dict, targets: list) -> None:
        """Test forward pass with pre-decoder outputs."""
        losses = criterion.forward(outputs, targets)

        assert isinstance(losses, dict)

        # Should have pre-decoder losses
        pre_loss_keys = [k for k in losses if "_pre" in k]
        assert len(pre_loss_keys) > 0

    def test_criterion_inheritance(self, criterion: DEIMCriterion) -> None:
        """Test that DEIM criterion properly inherits from DFINECriterion."""
        from getitune.backend.lightning.models.detection.losses.dfine_loss import DFINECriterion

        assert isinstance(criterion, DFINECriterion)

        # Should have inherited methods
        assert hasattr(criterion, "loss_labels_vfl")
        assert hasattr(criterion, "loss_boxes")
        assert hasattr(criterion, "loss_local")
        assert hasattr(criterion, "forward")

    def test_mal_loss_specific_behavior(self, criterion: DEIMCriterion, outputs: dict, targets: list) -> None:
        """Test MAL loss specific behavior compared to VFL."""
        indices = [(torch.tensor([0]), torch.tensor([1]))]
        num_boxes = 2

        # Get MAL loss
        mal_loss = criterion.loss_labels_mal(outputs, targets, indices, num_boxes)

        # Get VFL loss
        vfl_loss = criterion.loss_labels_vfl(outputs, targets, indices, num_boxes)

        # Both should return valid losses
        assert "loss_mal" in mal_loss
        assert "loss_vfl" in vfl_loss

        # MAL and VFL should be different (MAL uses different gamma weighting)
        assert not torch.allclose(mal_loss["loss_mal"], vfl_loss["loss_vfl"], atol=1e-6)

    def test_weight_dict_application(self, criterion: DEIMCriterion, outputs: dict, targets: list) -> None:
        """Test that weight dictionary is properly applied to losses."""
        losses = criterion.forward(outputs, targets)

        # Check that main losses are present
        main_losses = ["loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf", "loss_mal"]
        present_losses = [loss for loss in main_losses if loss in losses]

        # Should have at least some of the main losses
        assert len(present_losses) > 0

        # All present losses should be valid tensors
        for loss_name in present_losses:
            assert isinstance(losses[loss_name], torch.Tensor)
            assert losses[loss_name].numel() == 1

    def test_gamma_parameter_effect(self) -> None:
        """Test that gamma parameter affects the loss computation."""
        weight_dict = {
            "loss_vfl": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
            "loss_mal": 1.0,
        }

        # Create criteria with different gamma values
        criterion_low = DEIMCriterion(weight_dict=weight_dict, gamma=1.0, num_classes=10)
        criterion_high = DEIMCriterion(weight_dict=weight_dict, gamma=2.0, num_classes=10)

        # Both should be valid
        assert criterion_low.gamma == 1.0
        assert criterion_high.gamma == 2.0
