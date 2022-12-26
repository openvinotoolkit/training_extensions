_base_ = "./detector.py"

model = dict(
    type="YOLOX",
    backbone=dict(type="CSPDarknet", deepen_factor=0.33, widen_factor=0.375),
    neck=dict(type="YOLOXPAFPN", in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(type="YOLOXHead", num_classes=80, in_channels=96, feat_channels=96),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65), max_per_img=100),
)
load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/yolox_tiny_8x8.pth"
