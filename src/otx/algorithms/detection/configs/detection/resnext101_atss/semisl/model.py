"""Model configuration of ResNeXt101-ATSS model for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

_base_ = [
    "../../../../../../recipes/stages/detection/semisl.py",
    "../../../base/models/detector.py",
]

model = dict(
    super_type="MeanTeacher",
    pseudo_conf_thresh=0.25,
    unlabeled_loss_weights={"cls": 0.1, "bbox": 0.0, "obj": 0.0, "centerness": 0.0},
    filter_empty_annotations=True,
    type="CustomATSS",
    backbone=dict(
        type="ResNeXt",
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://resnext101_64x4d"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=5,
        relu_before_extra_convs=True,
    ),
    bbox_head=dict(
        type="CustomATSSHead",
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        use_qfl=False,
        qfl_cfg=dict(
            type="QualityFocalLoss",
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(
        assigner=dict(type="ATSSAssigner", topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)

load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/\
models/object_detection/v2/resnext101_atss_070623.pth"

fp16 = dict(loss_scale=512.0)
