dataset_type = 'CocoDataset'
img_size = (1024, 1024)

img_norm_cfg = dict(
    mean=(103.53, 116.28, 123.675), std=(1.0, 1.0, 1.0), to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True,
         with_mask=True, poly2mask=False),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(samples_per_gpu=4,
            workers_per_gpu=2,
            train=dict(
                type='RepeatDataset',
                adaptive_repeat_times=True,
                times=1,
                dataset=dict(
                    type=dataset_type,
                    ann_file='data/coco/annotations/instances_train2017.json',
                    img_prefix='data/coco/train2017',
                    pipeline=train_pipeline)),
            val=dict(
                type=dataset_type,
                test_mode=True,
                ann_file='data/coco/annotations/instances_val2017.json',
                img_prefix='data/coco/val2017',
                pipeline=test_pipeline),
            test=dict(
                type=dataset_type,
                test_mode=True,
                ann_file='data/coco/annotations/instances_val2017.json',
                img_prefix='data/coco/val2017',
                pipeline=test_pipeline))