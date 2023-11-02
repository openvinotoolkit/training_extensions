# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict

import pytest
from mmengine.config import Config


@pytest.fixture
def fxt_cfg_custom_atss(num_classes: int = 3) -> Dict:
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
            use_qfl=False,
            qfl_cfg=dict(type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0),
        ),
        train_cfg=train_cfg,
    )
    return cfg
