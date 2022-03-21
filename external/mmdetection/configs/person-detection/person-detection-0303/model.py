model = dict(
    type='ATSS',
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(2, 3, 4, 5),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=False,
        ),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=64,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=1,
        in_channels=64,
        stacked_convs=4,
        feat_channels=64,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

dataset_type = 'CocoDataset'
data_root = '../../data/airport/'
test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1280, 720),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False),
                dict(
                    type='Normalize',
                    mean=[0, 0, 0],
                    std=[255, 255, 255],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
]

train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='MinIoURandomCrop',
            min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
            min_crop_size=0.3),
        dict(
            type='Resize',
            img_scale=[(1280, 720), (896, 720), (1088, 720),
                        (1280, 672), (1280, 800)],
            multiscale_mode='value',
            keep_ratio=False),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[0, 0, 0],
            std=[255, 255, 255],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    samples_per_gpu=9,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            labels=('person',),
            ann_file=data_root + 'annotation_person_train.json',
            min_size=20,
            img_prefix=data_root + 'train',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        labels=('person',),
        ann_file=data_root + 'annotation_person_val.json',
        img_prefix=data_root + 'val',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        labels=('person',),
        ann_file=data_root + 'annotation_person_val.json',
        img_prefix=data_root + 'val',
        test_mode=True,
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='bbox_mAP',
    patience=5,
    iteration_patience=600,
    interval=1,
    min_lr=9e-06,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.3333333333333333)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
# yapf:enable
runner = dict(type='EpochRunnerWithCancel', max_epochs=10)
evaluation = dict(interval=1, metric='mAP', save_best='mAP')
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'output'
load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/person_detection_0303.pth'
resume_from = None
workflow = [('train', 1)]
