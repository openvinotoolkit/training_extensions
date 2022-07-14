dataset_type = 'CocoDataset'
img_size = (1344, 800)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True,
         with_mask=True, poly2mask=False),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_size,
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=False),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img'])
         ])
]

data = dict(samples_per_gpu=4,
            workers_per_gpu=2,
            train=dict(
                type=dataset_type,
                ann_file='data/coco/annotations/instances_train2017.json',
                img_prefix='data/coco/train2017',
                pipeline=train_pipeline),
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
