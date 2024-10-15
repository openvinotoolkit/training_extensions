# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of YOLOv9 criterion architecture."""

from unittest.mock import MagicMock, Mock

import pytest
import torch
from otx.algo.detection.losses.yolov9_loss import BCELoss, BoxLoss, BoxMatcher, DFLoss, YOLOv9Criterion, calculate_iou
from otx.algo.detection.utils.utils import Vec2Box
from torch import Tensor


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
    @pytest.fixture()
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
    @pytest.fixture()
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


class TestBoxMatcher:
    @pytest.fixture()
    def box_matcher(self) -> BoxMatcher:
        class_num = 10
        anchors = torch.tensor([[10, 10], [20, 20], [30, 30]])
        return BoxMatcher(class_num, anchors)

    def test_get_valid_matrix(self, box_matcher: BoxMatcher) -> None:
        target_bbox = torch.tensor([[[5, 5, 15, 15], [25, 25, 35, 35]]])

        valid_matrix = box_matcher.get_valid_matrix(target_bbox)

        expected_valid_matrix = torch.tensor([[[True, False, False], [False, False, True]]])
        assert torch.all(valid_matrix == expected_valid_matrix)

    def test_get_cls_matrix(self, box_matcher: BoxMatcher) -> None:
        predict_cls = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
        target_cls = torch.tensor([[[1, 2]]])

        cls_matrix = box_matcher.get_cls_matrix(predict_cls, target_cls)

        expected_cls_matrix = torch.tensor([[[0.2, 0.6]]])
        assert torch.all(cls_matrix == expected_cls_matrix)

    def test_get_iou_matrix(self, box_matcher: BoxMatcher) -> None:
        predict_bbox = torch.tensor([[[5.0, 5.0, 15.0, 15.0], [25.0, 25.0, 35.0, 35.0]]])
        target_bbox = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]])

        iou_matrix = box_matcher.get_iou_matrix(predict_bbox, target_bbox)

        expected_iou_matrix = torch.tensor([[[0.0317, 0.0000], [0.0000, 0.0317]]])
        assert torch.allclose(iou_matrix, expected_iou_matrix, atol=1e-4)

    def test_filter_topk(self, box_matcher: BoxMatcher) -> None:
        target_matrix = torch.tensor([[[0.5, 0.3, 0.7], [0.2, 0.4, 0.6]]])

        topk_targets, topk_masks = box_matcher.filter_topk(target_matrix, topk=2)

        expected_topk_targets = torch.tensor([[[0.5, 0.0, 0.7], [0.0, 0.4, 0.6]]])
        expected_topk_masks = torch.tensor([[[True, False, True], [False, True, True]]])
        assert torch.all(topk_targets == expected_topk_targets)
        assert torch.all(topk_masks == expected_topk_masks)

    def test_filter_duplicates(self, box_matcher: BoxMatcher) -> None:
        target_matrix = torch.tensor([[[0.5, 0.3, 0.7], [0.2, 0.4, 0.6]]])

        unique_indices = box_matcher.filter_duplicates(target_matrix)

        expected_unique_indices = torch.tensor([[[0], [1], [0]]])
        assert torch.all(unique_indices == expected_unique_indices)

    def test_call(self, box_matcher: BoxMatcher) -> None:
        target = torch.tensor([[[0, 0, 0, 1, 1], [0, 2, 2, 4, 4]]])
        predict = (torch.randint(0, 1, (1, 8400, 1)), torch.randn(1, 8400, 4))

        # Mock the necessary methods
        box_matcher.get_valid_matrix = MagicMock(
            return_value=torch.tensor([[[True, False, False], [False, False, True]]]),
        )
        box_matcher.get_iou_matrix = MagicMock(
            return_value=torch.tensor([[[0.0317, 0.0000, 1.0], [0.0000, 0.0317, 1.0]]]),
        )
        box_matcher.get_cls_matrix = MagicMock(return_value=torch.tensor([[[0.2, 0.6, 1.0]]]))
        box_matcher.filter_topk = MagicMock(
            return_value=(
                torch.tensor([[[0.5, 0.0, 0.7], [0.0, 0.4, 0.6]]]),
                torch.tensor([[[True, False, True], [False, True, True]]]),
            ),
        )
        box_matcher.filter_duplicates = MagicMock(return_value=torch.tensor([[[0], [1], [0]]]))

        align_targets, valid_masks = box_matcher(target, predict)

        assert align_targets.shape == torch.Size([1, 3, 14])
        assert valid_masks.shape == torch.Size([1, 3])

    def test_call_with_empty_bbox(self, box_matcher: BoxMatcher) -> None:
        target = torch.zeros((1, 0, 5))

        predict_cls = torch.rand((1, 8400, 10))
        predict_bbox = torch.rand((1, 8400, 4))
        predict = (predict_cls, predict_bbox)

        align_targets, valid_masks = box_matcher(target, predict)

        assert align_targets.shape == (1, 8400, 14)
        assert torch.all(align_targets == 0)

        assert valid_masks.shape == (1, 8400)
        assert torch.all(~valid_masks)


class TestYOLOv9Criterion:
    @pytest.fixture()
    def criterion(self) -> YOLOv9Criterion:
        num_classes = 10
        vec2box = Vec2Box(None, (640, 640), [8, 16, 32])
        return YOLOv9Criterion(
            num_classes,
            vec2box,
            loss_cls=Mock(return_value=torch.randn(1)),
            loss_dfl=Mock(return_value=torch.randn(1)),
            loss_iou=Mock(return_value=torch.randn(1)),
        )

    def test_forward(self, mocker, criterion: YOLOv9Criterion) -> None:
        mocker.patch.object(criterion, "vec2box", return_value=torch.tensor(0.0))
        mocker.patch.object(
            criterion,
            "_forward",
            return_value=(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)),
        )
        main_preds = torch.tensor(0.0)
        targets = torch.zeros(1, 1, 4)
        aux_preds = torch.tensor(0.0)

        loss_dict = criterion.forward(main_preds, targets, aux_preds)

        assert "loss_cls" in loss_dict
        assert "loss_df" in loss_dict
        assert "loss_iou" in loss_dict
        assert "total_loss" in loss_dict

    def test_separate_anchor(self, criterion: YOLOv9Criterion) -> None:
        criterion.num_classes = 2
        anchors = torch.cat((torch.zeros(1, 8400, 2), torch.ones(1, 8400, 4)), dim=-1)
        expected_cls = torch.zeros(1, 8400, 2)
        expected_box = torch.ones(1, 8400, 4) / criterion.vec2box.scaler[None, :, None]

        anchors_cls, anchors_box = criterion.separate_anchor(anchors)

        assert torch.allclose(anchors_cls, expected_cls)
        assert torch.allclose(anchors_box, expected_box)
