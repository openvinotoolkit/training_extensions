"""Unit tests of Custom Roi head for OTX template."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import numpy as np
from mmdet.registry import MODELS
from mmdet.structures.mask import PolygonMasks
from mmengine.config import Config

from otx.v2.adapters.torch.mmengine.mmdet.modules.models.heads.custom_roi_head import (
    CustomRoIHead,
    CustomConvFCBBoxHead
)


class TestCustomRoIHead:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = dict(
            type='CustomRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign',
                    output_size=7,
                    sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='L1Loss',
                    loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign',
                    output_size=14,
                    sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            mask_head=dict(
                type='CustomFCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=80,
                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_mask=True,
                    loss_weight=1.0)
            ),
            train_cfg=dict(
                assigner=dict(
                    type='CustomMaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=300),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False
            )
        )
        self.head = MODELS.build(Config(cfg))

    def test_init_bbox_head(self):
        assert isinstance(self.head, CustomRoIHead)
        assert isinstance(self.head.bbox_head, CustomConvFCBBoxHead)

    def test_loss(self):
        inpout_features = [
            torch.randn(1, 256, 256, 256),
            torch.randn(1, 256, 128, 128),
            torch.randn(1, 256, 64, 64),
            torch.randn(1, 256, 32, 32),
            torch.randn(1, 256, 16, 16),
        ]
        rpn_results_list = [
            Config(
                {
                    "bboxes": torch.tensor([[374.,  625.,  763.,  783.]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0], dtype=torch.int64),
                }
            )
        ]
        batch_data_samples = [
            Config(
                {
                    "gt_instances": Config(
                        {
                            "bboxes": torch.tensor([[374.,  625.,  763.,  783.]]),
                            "labels": torch.tensor([1], dtype=torch.int64),
                            "masks": PolygonMasks(
                                [
                                    [
                                        np.array(
                                            [
                                                359., 621., 358., 623., 358.,
                                                623., 357., 624., 357., 624.,
                                                356., 623., 358., 626., 359., 626.
                                            ]
                                        )
                                    ]
                                ],
                                1024,
                                1024,
                            )
                        }
                    ),
                    "metainfo": {"img_shape": (1024, 1024)},

                }
            )
        ]
        losses = self.head.loss(inpout_features, rpn_results_list, batch_data_samples)
        assert "loss_cls" in losses
        assert "acc" in losses
        assert "loss_bbox" in losses
        assert "loss_mask" in losses
