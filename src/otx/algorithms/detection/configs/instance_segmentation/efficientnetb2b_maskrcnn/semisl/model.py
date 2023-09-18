"""Model Configuration of Mask-RCNN EfficientNetb2b model for Semi-Supervised Learning Instance Segmentation Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

_base_ = [
    "../../../../../../recipes/stages/instance-segmentation/semisl.py",
    "../../../../../common/adapters/mmcv/configs/backbones/efficientnet_b2b.yaml",
    "../../../base/models/detector.py",
]

task = "instance-segmentation"

model = dict(
    super_type="MeanTeacher",
    pseudo_conf_thresh=0.7,
    unlabeled_loss_weights={"cls": 2.0, "bbox": 1.0, "mask": 1.0},
    type="CustomMaskRCNN",
    neck=dict(type="FPN", in_channels=[24, 48, 120, 352], out_channels=80, num_outs=5),
    rpn_head=dict(
        type="RPNHead",
        in_channels=80,
        feat_channels=80,
        anchor_generator=dict(type="AnchorGenerator", scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="CustomRoIHead",  # Use CustomROIHead for Ignore mode
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=80,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=80,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="OrdinaryFocalLoss", gamma=1.5, loss_weight=1.0),
            loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=80,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=dict(
            type="CustomFCNMaskHead",
            num_convs=4,
            in_channels=80,
            conv_out_channels=80,
            num_classes=80,
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
            sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.8),
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
            sampler=dict(type="RandomSampler", num=256, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=800,
            max_per_img=500,
            nms=dict(type="nms", iou_threshold=0.8),
            min_bbox_size=0,
        ),
        rcnn=dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.7), max_per_img=500, mask_thr_binary=0.5),
    ),
)
load_from = "https://storage.openvinotoolkit.org/repositories/\
openvino_training_extensions/models/instance_segmentation/\
v2/efficientnet_b2b-mask_rcnn-576x576.pth"

evaluation = dict(interval=1, metric="mAP", save_best="mAP", iou_thr=[0.5])
fp16 = dict(loss_scale=512.0)
ignore = True
