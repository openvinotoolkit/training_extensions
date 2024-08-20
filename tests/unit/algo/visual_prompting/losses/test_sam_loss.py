# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from otx.algo.visual_prompting.losses.sam_loss import SAMCriterion
from torch import Tensor


class TestSAMCriterion:
    @pytest.fixture()
    def sam_criterion(self) -> SAMCriterion:
        return SAMCriterion(1024)

    def test_forward(self, mocker, sam_criterion: SAMCriterion) -> None:
        """Test forward method."""
        pred_masks = [
            Tensor([[0, 0, 0.5, 0.5, 0, 0]]),
            Tensor([[0, 0, 0.3, 0.3, 0, 0]]),
        ]
        gt_masks = [
            Tensor([[0, 0, 1, 1, 0, 0]]),
            Tensor([[0, 0, 1, 1, 0, 0]]),
        ]
        ious = [
            Tensor([0.5]),
            Tensor([0.7]),
        ]
        ori_shapes = [
            Tensor([1024, 1024]),
            Tensor([1024, 1024]),
        ]
        expected_loss = torch.tensor(1.0146)
        expected_loss_focal = torch.tensor(0.0163)
        expected_loss_dice = torch.tensor(0.3194)
        expected_loss_iou = torch.tensor(0.3700)

        mocker.patch("otx.algo.visual_prompting.losses.sam_loss.postprocess_masks", lambda x, y, z: x)  # noqa: ARG005
        mocker.patch("torch.Tensor.sigmoid", lambda x: x)
        mocker.patch("torch.Tensor.flatten", lambda x, y: x)  # noqa: ARG005

        results = sam_criterion.forward(pred_masks, gt_masks, ious, ori_shapes)

        atol = 1e-04
        rtol = 1e-03
        assert torch.isclose(results["loss"], expected_loss, atol=atol, rtol=rtol)
        assert torch.isclose(results["loss_focal"], expected_loss_focal, atol=atol, rtol=rtol)
        assert torch.isclose(results["loss_dice"], expected_loss_dice, atol=atol, rtol=rtol)
        assert torch.isclose(results["loss_iou"], expected_loss_iou, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("inputs", "targets", "expected"),
        [
            (Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0])),
            (Tensor([[0, 0, 0.5, 0.5, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.25])),
            (Tensor([[0, 0, 0.3, 0.3, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.3888888359])),
        ],
    )
    def test_calculate_dice_loss(
        self,
        sam_criterion: SAMCriterion,
        inputs: Tensor,
        targets: Tensor,
        expected: Tensor,
    ) -> None:
        """Test calculate_dice_loss."""
        results = sam_criterion.calculate_dice_loss(inputs, targets, num_masks=1)

        assert torch.isclose(results, expected)

    @pytest.mark.parametrize(
        ("inputs", "targets", "expected"),
        [
            (Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0])),
            (Tensor([[0, 0, 0.5, 0.5, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0098766042])),
            (Tensor([[0, 0, 0.3, 0.3, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0226361733])),
        ],
    )
    def test_calculate_sigmoid_ce_focal_loss(
        self,
        sam_criterion: SAMCriterion,
        inputs: Tensor,
        targets: Tensor,
        expected: Tensor,
    ) -> None:
        """Test calculate_sigmoid_ce_focal_loss."""
        results = sam_criterion.calculate_sigmoid_ce_focal_loss(inputs, targets, num_masks=1)

        assert torch.isclose(results, expected)

    @pytest.mark.parametrize(
        ("inputs", "targets", "expected"),
        [
            (Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([1.0])),
            (Tensor([[0, 0, 0.5, 0.5, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([1.0])),
            (Tensor([[0, 0, 0.3, 0.3, 0, 0]]), Tensor([[0, 0, 1, 1, 0, 0]]), Tensor([0.0])),
        ],
    )
    def test_calculate_iou(
        self,
        sam_criterion: SAMCriterion,
        inputs: Tensor,
        targets: Tensor,
        expected: Tensor,
    ) -> None:
        """Test calculate_iou."""
        results = sam_criterion.calculate_iou(inputs, targets)

        assert results == expected
