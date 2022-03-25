dataset_type = 'CustomDataset'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

img_scale = (544, 544)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MaskCompose', prob=0.5, lambda_limits=(4, 16), keep_original=False,
         transforms=[
             dict(type='PhotoMetricDistortion'),
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='RandomRotate', prob=0.5, degree=30, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset',
        adaptive_repeat=True,
        times=1,
        dataset=dict(
            type=dataset_type,
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline
    )
)
