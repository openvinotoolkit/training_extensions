# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit test of RTMDetInstCriterion of OTX Instance Segmentation tasks."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from otx.algo.common.losses import GIoULoss, QualityFocalLoss
from otx.algo.instance_segmentation.losses import DiceLoss, RTMDetInstCriterion


class TestRTMDetInstCriterion:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"batch_pos_mask_logits": torch.randn(3, 160, 160), "pos_gt_masks": torch.zeros(3, 160, 160), "num_pos": 3},
            {"zero_loss": torch.tensor(0.0), "num_pos": 0},
        ],
    )
    def test_forward(self, mocker, kwargs: dict[str, Any]) -> None:
        mocker.patch(
            "otx.algo.instance_segmentation.losses.rtmdet_inst_loss.RTMDetCriterion.forward",
            return_value={"loss_cls": 0, "loss_bbox": 0},
        )

        criterion = RTMDetInstCriterion(
            num_classes=3,
            loss_cls=QualityFocalLoss(
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0,
            ),
            loss_bbox=GIoULoss(loss_weight=2.0),
            loss_mask=DiceLoss(
                loss_weight=2.0,
                eps=5.0e-06,
                reduction="mean",
            ),
        )

        loss = criterion(
            cls_score=0,
            bbox_pred=0,
            labels=0,
            label_weights=0,
            bbox_targets=0,
            assign_metrics=0,
            stride=0,
            **kwargs,
        )
        assert "loss_cls" in loss
        assert "loss_bbox" in loss
        assert "loss_mask" in loss

        if kwargs["num_pos"] == 0:
            assert loss["loss_mask"] == 0
