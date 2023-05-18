dataset_type = 'CocoDataset'
img_size = (992, 736)
tile_cfg = dict(
    tile_size=400,
    min_area_ratio=0.9,
    overlap_ratio=0.2,
    iou_threshold=0.45,
    max_per_img=1500,
    filter_empty_gt=True)
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672),
                   (992, 800)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=(992, 736),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255, 255, 255],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
train_dataset = dict(
    type='ImageTilingDataset',
    dataset=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_train.json',
        img_prefix='data/coco/images/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    pipeline=[
        dict(
            type='MinIoURandomCrop',
            min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
            min_crop_size=0.3),
        dict(
            type='Resize',
            img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672),
                       (992, 800)],
            multiscale_mode='value',
            keep_ratio=False),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize', mean=[0, 0, 0], std=[255, 255, 255],
            to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    tile_size=400,
    min_area_ratio=0.9,
    overlap_ratio=0.2,
    iou_threshold=0.45,
    max_per_img=1500,
    filter_empty_gt=True)
val_dataset = dict(
    type='ImageTilingDataset',
    dataset=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val.json',
        img_prefix='data/coco/images/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
        ]),
    pipeline=[
        dict(
            type='MultiScaleFlipAug',
            img_scale=(992, 736),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[0, 0, 0],
                    std=[255, 255, 255],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ],
    tile_size=400,
    min_area_ratio=0.9,
    overlap_ratio=0.2,
    iou_threshold=0.45,
    max_per_img=1500,
    filter_empty_gt=True)
test_dataset = dict(
    type='ImageTilingDataset',
    dataset=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_test.json',
        img_prefix='data/coco/images/test',
        test_mode=True,
        pipeline=[dict(type='LoadImageFromFile')]),
    pipeline=[
        dict(
            type='MultiScaleFlipAug',
            img_scale=(992, 736),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[0, 0, 0],
                    std=[255, 255, 255],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ],
    tile_size=400,
    min_area_ratio=0.9,
    overlap_ratio=0.2,
    iou_threshold=0.45,
    max_per_img=1500,
    filter_empty_gt=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ImageTilingDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file='data/coco/annotations/instances_train.json',
            img_prefix='data/coco/images/train',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ]),
        pipeline=[
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(
                type='Resize',
                img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672),
                           (992, 800)],
                multiscale_mode='value',
                keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255, 255, 255],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        tile_size=400,
        min_area_ratio=0.9,
        overlap_ratio=0.2,
        iou_threshold=0.45,
        max_per_img=1500,
        filter_empty_gt=True),
    val=dict(
        type='ImageTilingDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file='data/coco/annotations/instances_val.json',
            img_prefix='data/coco/images/val',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
            ]),
        pipeline=[
            dict(
                type='MultiScaleFlipAug',
                img_scale=(992, 736),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        tile_size=400,
        min_area_ratio=0.9,
        overlap_ratio=0.2,
        iou_threshold=0.45,
        max_per_img=1500,
        filter_empty_gt=True),
    test=dict(
        type='ImageTilingDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file='data/coco/annotations/instances_test.json',
            img_prefix='data/coco/images/test',
            test_mode=True,
            pipeline=[dict(type='LoadImageFromFile')]),
        pipeline=[
            dict(
                type='MultiScaleFlipAug',
                img_scale=(992, 736),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        tile_size=400,
        min_area_ratio=0.9,
        overlap_ratio=0.2,
        iou_threshold=0.45,
        max_per_img=1500,
        filter_empty_gt=True))
