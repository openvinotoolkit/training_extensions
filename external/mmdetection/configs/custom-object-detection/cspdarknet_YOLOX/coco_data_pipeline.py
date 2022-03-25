dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (640, 640)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=(416, 416), pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file='data/coco/annotations/instances_train2017.json',
            img_prefix='data/coco/train2017',
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
        # filter_empty_gt=False,
        ),
    pipeline=train_pipeline,
    dynamic_scale=img_scale),
    val=dict(
        type=dataset_type,
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017',
        pipeline=test_pipeline))
