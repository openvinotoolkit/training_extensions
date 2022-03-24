_base_ = [
    './coco_data_pipeline.py'
]

model = dict(
    type='YOLOX',
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.375),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=80,
        in_channels=96,
        feat_channels=96),
    train_cfg=dict(
        assigner=dict(
            type='SimOTAAssigner',
            center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=100))

evaluation = dict(interval=1, metric='mAP', save_best='mAP')
optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='mAP',
    patience=3,
    iteration_patience=300,
    interval=1,
    min_lr=0.000001,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001
)

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
runner = dict(type='EpochRunnerWithCancel', max_epochs=300)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'output'
load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/yolox_tiny_8x8.pth'
resume_from = None
workflow = [('train', 1)]

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        patience=10,
        iteration_patience=1000,
        metric='mAP',
        interval=1,
        priority=75),
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=15,
        priority=48),
    dict(
        type='SyncRandomSizeHook',
        ratio_range=(10, 20),
        img_scale=(640, 640),
        interval=1,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=15,
        interval=1,
        priority=48),
]




