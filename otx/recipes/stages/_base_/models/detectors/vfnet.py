_base_ = "./detector.py"

model = dict(
    type="VFNet",
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
        type="VFNetHead",
        num_classes=2,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type="VarifocalLoss", use_sigmoid=True, alpha=0.75, gamma=2.0, iou_weighted=True, loss_weight=1.0
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=1.5),
        loss_bbox_refine=dict(type="GIoULoss", loss_weight=2.0),
    ),
    train_cfg=dict(assigner=dict(type="ATSSAssigner", topk=9), allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(
        nms_pre=1000, min_bbox_size=0, score_thr=0.01, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100
    ),
)
load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/resnet50-vfnet.pth"
