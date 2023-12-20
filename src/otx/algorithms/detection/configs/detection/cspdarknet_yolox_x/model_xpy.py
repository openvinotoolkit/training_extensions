"""Model configuration of YOLOX_X model for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

_base_ = ["../../../../../recipes/stages/detection/incremental.py", "../../base/models/detector.py"]

model = dict(
    type="CustomYOLOX",
    backbone=dict(type="CSPDarknet", deepen_factor=1.33, widen_factor=1.25, out_indices=(2, 3, 4)),
    neck=dict(type="YOLOXPAFPN", in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(type="CustomYOLOXHead", num_classes=80, in_channels=320, feat_channels=320),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65), max_per_img=100),
    size_multiplier=160,
    random_size_range=(3, 5),
)
load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolox\
/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"

fp16 = dict(loss_scale=512.0, bf16_training=False)
ignore = False
