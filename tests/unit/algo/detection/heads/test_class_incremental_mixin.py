# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of ClassIncrementalMixin."""

import torch
from otx.algo.common.losses import CrossEntropyLoss, CrossSigmoidFocalLoss, GIoULoss
from otx.algo.common.utils.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from otx.algo.detection.atss import ATSS
from otx.algo.detection.losses import ATSSCriterion


class MockGTInstance:
    bboxes = torch.Tensor([[0.0, 0.0, 240, 240], [240, 240, 480, 480]])
    labels = torch.LongTensor([0, 1])


class TestClassIncrementalMixin:
    def test_ignore_label(self, mocker) -> None:
        atss = ATSS(model_name="atss_mobilenetv2", label_info=3, input_size=(800, 992))
        criterion = ATSSCriterion(
            num_classes=3,
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            loss_cls=CrossSigmoidFocalLoss(
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_bbox=GIoULoss(loss_weight=2.0),
            loss_centerness=CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0),
        )
        atss_head = atss.model.bbox_head

        backbone_features = (
            torch.randn(1, 64, 100, 124),
            torch.randn(1, 64, 50, 62),
            torch.randn(1, 64, 25, 31),
            torch.randn(1, 64, 13, 16),
            torch.randn(1, 64, 7, 8),
        )

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

        mocker.patch(
            "otx.algo.detection.heads.base_head.unpack_det_entity",
            return_value=(batch_gt_instances, batch_img_metas),
        )
        loss_with_ignored_labels = criterion(
            **atss_head.prepare_loss_inputs(
                x=backbone_features,
                entity=mocker.MagicMock(),
            ),
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

        mocker.patch(
            "otx.algo.detection.heads.base_head.unpack_det_entity",
            return_value=(batch_gt_instances, batch_img_metas),
        )
        loss_without_ignored_labels = criterion(
            **atss_head.prepare_loss_inputs(
                x=backbone_features,
                entity=mocker.MagicMock(),
            ),
        )
        loss_cls_without_ignored_labels = torch.sum(torch.Tensor(loss_without_ignored_labels["loss_cls"]))

        assert loss_cls_with_ignored_labels < loss_cls_without_ignored_labels
