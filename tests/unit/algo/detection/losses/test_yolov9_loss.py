# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of YOLOv9 criterion architecture."""

import pytest
import torch
from torch import Tensor

from otx.algo.detection.losses.yolov9_loss import BCELoss, calculate_iou
import torch
from otx.algo.detection.losses.yolov9_loss import BoxLoss
import torch
import pytest
from otx.algo.detection.losses.yolov9_loss import DFLoss
from otx.algo.detection.utils.yolov7_v9_utils import Vec2Box


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ("iou", torch.tensor([[0.1429, 0.0000], [1.0000, 0.1429]])),
        ("diou", torch.tensor([[0.0317, -0.2500], [1.0000, 0.0317]])),
        ("ciou", torch.tensor([[0.0317, -0.2500], [1.0000, 0.0317]])),
    ],
)
def test_calculate_iou(metrics: str, expected: Tensor) -> None:
    bbox1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    bbox2 = torch.tensor([[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 4.0, 4.0]])

    iou = calculate_iou(bbox1, bbox2, metrics=metrics)

    assert torch.allclose(iou, expected, atol=1e-4)


class TestBCELoss:
    def test_forward(self) -> None:
        loss_fn = BCELoss()
        predicts_cls = torch.tensor([[0.5, 0.8, 0.2], [0.3, 0.6, 0.9]])
        targets_cls = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        cls_norm = torch.tensor(6.0)

        loss = loss_fn(predicts_cls, targets_cls, cls_norm)

        expected_loss = torch.tensor(0.7961)
        assert torch.allclose(loss, expected_loss, atol=1e-4)


class TestBoxLoss:
    @pytest.fixture
    def box_loss(self) -> BoxLoss:
        return BoxLoss()

    @pytest.mark.parametrize(
        ("box_norm", "expected"),
        [
            (torch.ones(1), torch.tensor(8400.0)),
            (torch.zeros(1), torch.tensor(0.0)),
        ],
    )
    def test_forward(self, box_loss: BoxLoss, box_norm: Tensor, expected: Tensor) -> None:
        predicts_bbox = torch.ones(1, 8400, 4)
        targets_bbox = torch.ones(1, 8400, 4)
        valid_masks = torch.ones(1, 8400, dtype=torch.bool)
        cls_norm = torch.ones(1)

        loss = box_loss(predicts_bbox, targets_bbox, valid_masks, box_norm, cls_norm)

        assert torch.allclose(loss, expected, atol=1e-4)


class TestDFLoss:
    @pytest.fixture
    def df_loss(self) -> DFLoss:
        return DFLoss(vec2box=Vec2Box(None, (640, 640), [8, 16, 32]))

    @pytest.mark.parametrize(
        ("box_norm", "expected"),
        [
            (torch.ones(1), torch.tensor(23289.7520)),
            (torch.zeros(1), torch.tensor(0.0)),
        ],
    )
    def test_forward(self, df_loss: DFLoss, box_norm: Tensor, expected: Tensor) -> None:
        predicts_anc = torch.ones(1, 8400, 4, 16)
        targets_bbox = torch.ones(1, 8400, 4)
        valid_masks = torch.ones(1, 8400, dtype=torch.bool)
        cls_norm = torch.ones(1)

        loss = df_loss.forward(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        assert torch.allclose(loss, expected, atol=1e-4)
