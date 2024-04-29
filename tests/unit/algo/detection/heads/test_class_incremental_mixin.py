# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of ClassIncrementalMixin."""

import torch
from otx.algo.detection.atss import MobileNetV2ATSS


class MockGTInstance:
    bboxes = torch.Tensor([[0.0, 0.0, 240, 240], [240, 240, 480, 480]])
    labels = torch.LongTensor([0, 1])


class TestClassIncrementalMixin:
    def test_ignore_label(self) -> None:
        atss = MobileNetV2ATSS(3)
        atss_head = atss.model.bbox_head

        cls_scores = [
            torch.randn(1, 3, 92, 124),
            torch.randn(1, 3, 46, 62),
            torch.randn(1, 3, 23, 31),
            torch.randn(1, 3, 12, 16),
            torch.randn(1, 3, 6, 8),
        ]
        bbox_preds = [
            torch.randn(1, 4, 92, 124),
            torch.randn(1, 4, 46, 62),
            torch.randn(1, 4, 23, 31),
            torch.randn(1, 4, 12, 16),
            torch.randn(1, 4, 6, 8),
        ]
        centernesses = [
            torch.randn(1, 1, 92, 124),
            torch.randn(1, 1, 46, 62),
            torch.randn(1, 1, 23, 31),
            torch.randn(1, 1, 12, 16),
            torch.randn(1, 1, 6, 8),
        ]

        batch_gt_instances = [MockGTInstance()]
        batch_img_metas = [
            {
                "ignored_labels": [2],
                "img_shape": (480, 480),
                "ori_shape": (480, 480),
                "scale_factor": (1.0, 1.0),
                "pad_shape": (480, 480),
            },
        ]

        loss_with_ignored_labels = atss_head.loss_by_feat(
            cls_scores,
            bbox_preds,
            centernesses,
            batch_gt_instances,
            batch_img_metas,
        )
        loss_cls_with_ignored_labels = torch.sum(torch.Tensor(loss_with_ignored_labels["loss_cls"]))

        batch_img_metas = [
            {
                "ignored_labels": None,
                "img_shape": (480, 480),
                "ori_shape": (480, 480),
                "scale_factor": (1.0, 1.0),
                "pad_shape": (480, 480),
            },
        ]
        loss_without_ignored_labels = atss_head.loss_by_feat(
            cls_scores,
            bbox_preds,
            centernesses,
            batch_gt_instances,
            batch_img_metas,
        )
        loss_cls_without_ignored_labels = torch.sum(torch.Tensor(loss_without_ignored_labels["loss_cls"]))

        assert loss_cls_with_ignored_labels < loss_cls_without_ignored_labels
