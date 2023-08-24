"""Model configuration of RTMDet-Inst model for Instance-Seg Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

_base_ = [
    "../../../../../recipes/stages/instance-segmentation/incremental.py",
    "../../base/models/detector.py",
]

task = "instance-segmentation"


model = dict(
    type="CustomRTMDetInst",
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    neck=dict(
        type="CSPNeXtPAFPN",
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    bbox_head=dict(
        type="RTMDetInsSepBNHead",
        num_classes=80,
        in_channels=256,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        feat_channels=256,
        act_cfg=dict(type="SiLU", inplace=True),
        norm_cfg=dict(type="BN", requires_grad=True),
        anchor_generator=dict(type="MlvlPointGenerator", offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type="DistancePointBBoxCoder"),
        loss_cls=dict(type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        loss_mask=dict(type="DiceLoss", loss_weight=2.0, eps=5e-6, reduction="mean"),
    ),
    train_cfg=dict(
        assigner=dict(type="DynamicSoftLabelAssigner", topk=13), allowed_border=-1, pos_weight=-1, debug=False
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5,
    ),
)

evaluation = dict(interval=1, metric="mAP", save_best="mAP", iou_thr=[0.5])

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.004,
    weight_decay=0.05,
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

lr_config = dict(min_lr=4e-06)

optimizer_config = dict(_delete_=True, grad_clip=None)

load_from = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"

ignore = True
