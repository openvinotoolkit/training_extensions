"""Rotated FCOS R50."""
_base_ = [
    "../../../../../recipes/stages/detection/incremental.py",
    "../../../../common/adapters/mmcv/configs/backbones/resnet50.yaml",
    "../../base/models/detector.py",
]

task = "mmrotate"
angle_version = "le90"

model = dict(
    type="CustomRotatedFCOS",
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
        type="RotatedFCOSHead",
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=True,
        scale_angle=True,
        bbox_coder=dict(type="DistanceAnglePointCoder", angle_version=angle_version),
        h_bbox_coder=dict(type="DistancePointBBoxCoder"),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="GIoULoss", loss_weight=1.0),
        loss_angle=dict(type="L1Loss", loss_weight=0.2),
        loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    ),
    train_cfg=None,
    test_cfg=dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05, nms=dict(iou_thr=0.1), max_per_img=2000),
)

load_from = "https://download.openmmlab.com/mmrotate/\
v0.1.0/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90/\
rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth"

optimizer = dict(type="SGD", lr=0.0025, momentum=0.9, weight_decay=0.0001)
