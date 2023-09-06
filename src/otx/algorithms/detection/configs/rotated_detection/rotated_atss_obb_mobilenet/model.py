"""Model of rotated_atss_obb_mobilenet for Rotated-Detection Task."""
_base_ = [
    "../../../../../recipes/stages/detection/incremental.py",
    "../../../../common/adapters/mmcv/configs/backbones/mobilenet_v2_w1.yaml",
    "../../base/models/detector.py",
]

task = "mmrotate"
angle_version = "le135"

model = dict(
    type="CustomRotatedRetinaNet",
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
        type="CustomRotatedATSSHead",
        num_classes=15,
        in_channels=64,
        stacked_convs=4,
        feat_channels=64,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type="RotatedAnchorGenerator",
            octave_base_scale=4,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHAOBBoxCoder",
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
        ),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type="CustomATSSObbAssigner", topk=9, angle_version=angle_version, iou_calculator=dict(type="RBboxOverlaps2D")
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05, nms=dict(iou_thr=0.1), max_per_img=2000),
)

load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions\
/models/object_detection/v2/mobilenet_v2-atss.pth"

fp16 = dict(loss_scale=512.0)
