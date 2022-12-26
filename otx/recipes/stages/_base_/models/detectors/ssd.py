_base_ = './single_stage_detector.py'

__input_size = 300

model = dict(
    bbox_head=dict(
        type='SSDHead',
        num_classes=80,
        in_channels=(512, 1024, 512, 256, 256, 256),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=__input_size,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]
        )
    ),
    train_cfg=dict(
        assigner=dict(
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
        )
    )
)
