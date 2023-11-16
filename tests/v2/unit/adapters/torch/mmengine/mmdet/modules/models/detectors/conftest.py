"""Fixtures for Unit tests of otx/v2/adapters/torch/mmengine/mmdet/modules/detectors."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import pytest
from mmengine.config import Config


@pytest.fixture
def fxt_cfg_custom_atss(num_classes: int = 3) -> dict:
    train_cfg = Config(
        dict(
            assigner=dict(type="ATSSAssigner", topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        )
    )
    cfg = dict(
        type="CustomATSS",
        backbone=dict(
            avg_down=False,
            base_channels=64,
            conv_cfg=None,
            dcn=None,
            deep_stem=False,
            depth=18,
            dilations=(1, 1, 1, 1),
            frozen_stages=-1,
            in_channels=3,
            init_cfg=None,
            norm_cfg=dict(requires_grad=True, type="BN"),
            norm_eval=True,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            plugins=None,
            pretrained=None,
            stage_with_dcn=(False, False, False, False),
            stem_channels=None,
            strides=(1, 2, 2, 2),
            style="pytorch",
            type="mmdet.ResNet",
            with_cp=False,
            zero_init_residual=True,
        ),
        neck=dict(
            type="FPN",
            in_channels=[64, 128, 256, 512],
            out_channels=64,
            start_level=1,
            add_extra_convs="on_output",
            num_outs=5,
            relu_before_extra_convs=True,
        ),
        bbox_head=dict(
            type="CustomATSSHead",
            num_classes=num_classes,
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
        ),
        train_cfg=train_cfg,
    )
    return cfg


@pytest.fixture
def fxt_cfg_custom_maskrcnn(num_classes: int = 3) -> dict:
    cfg = dict(
        type="CustomMaskRCNN",
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            style='pytorch',
            frozen_stages=1,
            norm_cfg=dict(
                type='BN',
                requires_grad=True),
            norm_eval=True),
        data_preprocessor=dict(
            type="DetDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32,
        ),
        neck=dict(
            type="FPN",
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
        ),
        rpn_head=dict(
            type="RPNHead",
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type="AnchorGenerator",
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
            ),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
        roi_head=dict(
            type="CustomRoIHead",  # Use CustomROIHead for Ignore mode
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32],
            ),
            bbox_head=dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=False,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="L1Loss", loss_weight=1.0),
            ),
            mask_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32],
            ),
            mask_head=dict(
                type="CustomFCNMaskHead",
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=num_classes,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
        ),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type="CustomMaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=300,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(
                nms_across_levels=False,
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                assigner=dict(
                    type="CustomMaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=300,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            ),
        ),
        test_cfg=dict(
            rpn=dict(
                nms_across_levels=False,
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type="nms", iou_threshold=0.5, max_num=100),
                max_per_img=100,
                mask_thr_binary=0.5,
            ),
        ),
    )
    return cfg
