# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from mmengine.config import Config

from otx.v2.adapters.torch.mmengine.mmdet.modules.models.heads import (
    CustomATSSHead,
)


class TestCrossDatasetDetectorHead:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.head = CustomATSSHead(
            num_classes=3,
            in_channels=64,
            stacked_convs=4,
            feat_channels=64,
            anchor_generator=dict(
                type="AnchorGenerator",
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            use_qualified_focal_loss=False,
            qualified_focal_loss_cfg=dict(type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0),
        )

    def test_get_atss_targets(self, mocker) -> None:
        anchors = torch.randn(5456, 4)
        labels = torch.randint(0, 3, (5456,))
        label_weights = torch.randn(5456)
        bbox_targets = torch.randn(5456, 4)
        bbox_weights = torch.randn(5456, 4)
        pos_inds_list = label_weights > 0.5
        neg_inds_list = label_weights <= 0.5
        sampling_results_list = Config({"avg_factor": 4})
        mocker.patch.object(
            CustomATSSHead,
            "_get_targets_single",
            return_value=(
                anchors,
                labels,
                label_weights,
                bbox_targets,
                bbox_weights,
                pos_inds_list,
                neg_inds_list,
                sampling_results_list,
            ),
        )

        anchor_list = [
            [
                torch.randn(4096, 4),
                torch.randn(1024, 4),
                torch.randn(256, 4),
                torch.randn(64, 4),
                torch.randn(16, 4),
            ]
        ] * 6
        valid_flag_list = [
            [
                torch.randn(4096),
                torch.randn(1024),
                torch.randn(256),
                torch.randn(64),
                torch.randn(16),
            ]
        ] * 6
        batch_img_metas = [{}] * 6
        batch_gt_instances = [{}] * 6
        out = self.head.get_atss_targets(anchor_list, valid_flag_list, batch_gt_instances, batch_img_metas)
        assert len(out) == 7

    def test_loss_by_feat(self, mocker):
        anchor_list = [
            torch.randn(6, 4096, 4),
            torch.randn(6, 1024, 4),
            torch.randn(6, 256, 4),
            torch.randn(6, 64, 4),
            torch.randn(6, 16, 4),
        ]
        labels_list = [
            torch.randint(0, 3, (6, 4096)),
            torch.randint(0, 3, (6, 1024)),
            torch.randint(0, 3, (6, 256)),
            torch.randint(0, 3, (6, 64)),
            torch.randint(0, 3, (6, 16)),
        ]
        label_weights_list = [
            torch.randn(6, 4096),
            torch.randn(6, 1024),
            torch.randn(6, 256),
            torch.randn(6, 64),
            torch.randn(6, 16),
        ]
        bbox_targets_list = [
            torch.randn(6, 4096, 4),
            torch.randn(6, 1024, 4),
            torch.randn(6, 256, 4),
            torch.randn(6, 64, 4),
            torch.randn(6, 16, 4),
        ]
        bbox_weights_list = [
            torch.randn(6, 4096, 4),
            torch.randn(6, 1024, 4),
            torch.randn(6, 256, 4),
            torch.randn(6, 64, 4),
            torch.randn(6, 16, 4),
        ]
        valid_label_mask = [
            torch.randn(6, 4096, 3),
            torch.randn(6, 1024, 3),
            torch.randn(6, 256, 3),
            torch.randn(6, 64, 3),
            torch.randn(6, 16, 3),
        ]

        mocker.patch.object(
            CustomATSSHead,
            "get_targets",
            return_value=[
                anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                19,
                valid_label_mask,
            ]
        )
        mocker.patch.object(
            CustomATSSHead,
            "get_anchors",
            return_value=([], []),
        )

        def mock_centerness_target(anchors, targets):
            return torch.randn(anchors.shape[0])

        mocker.patch.object(
            CustomATSSHead,
            "centerness_target",
            side_effect=mock_centerness_target,
        )

        cls_scores = [
            torch.randn(6, 3, 64, 64),
            torch.randn(6, 3, 32, 32),
            torch.randn(6, 3, 16, 16),
            torch.randn(6, 3, 8, 8),
            torch.randn(6, 3, 4, 4),
        ]
        bbox_preds = [
            torch.randn(6, 4, 64, 64),
            torch.randn(6, 4, 32, 32),
            torch.randn(6, 4, 16, 16),
            torch.randn(6, 4, 8, 8),
            torch.randn(6, 4, 4, 4),
        ]
        centernesses = [
            torch.randn(6, 1, 64, 64),
            torch.randn(6, 1, 32, 32),
            torch.randn(6, 1, 16, 16),
            torch.randn(6, 1, 8, 8),
            torch.randn(6, 1, 4, 4),
        ]

        outs = self.head.loss_by_feat(cls_scores, bbox_preds, centernesses, [{}] * 6, [{}] * 6)
        assert "loss_cls" in outs
        assert "loss_bbox" in outs
        assert "loss_centerness" in outs
