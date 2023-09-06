"""Rotated FCOS with MobileNetV2 backbone."""
_base_ = [
    "../../../../../recipes/stages/detection/incremental.py",
    "../../../../common/adapters/mmcv/configs/backbones/mobilenet_v2_w1.yaml",
    "../../base/models/detector.py",
]

task = "mmrotate"
angle_version = "le90"

model = dict(
    type="CustomRotatedFCOS",
    backbone=dict(out_indices=[2, 3, 4, 5]),
    neck=dict(
        type="FPN",
        in_channels=[24, 32, 96, 320],
        out_channels=64,
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5,
        relu_before_extra_convs=True,
    ),
    bbox_head=dict(
        type="RotatedFCOSHead",
        num_classes=15,
        in_channels=64,
        stacked_convs=4,
        feat_channels=64,
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

# TODO[EUGENE]: Add pretrained weights
fp16 = dict(loss_scale=512.0)
