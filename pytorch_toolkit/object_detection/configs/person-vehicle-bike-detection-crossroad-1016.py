# model settings
input_size = 512
model = dict(
    type='SingleStageDetector',
    pretrained=None,
    backbone=dict(
        type='SSDMobilenetV2',
        input_size=input_size,
        scales=6,
        ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        input_size=input_size,
        in_channels=(576, 1280, 512),
        num_classes=4,
        anchor_strides=(16, 32, 64),
        anchor_widths=([17.137, 38.165, 70.69, 9.584, 17.634, 23.744, 6.507, 12.245, 14.749],
                       [81.753, 153.183, 169.567, 32.148, 41.048, 52.198, 32.391, 22.397, 33.216],
                       [110.651, 237.237, 348.269, 65.598, 82.729, 110.538, 53.24, 68.246, 105.444],
                       ),
        anchor_heights=([20.733, 45.464, 78.592, 29.393, 55.398, 84.88, 17.006, 28.673, 44.11],
                        [157.379, 104.698, 210.545, 118.319, 157.328, 203.363, 36.256, 64.451, 101.718],
                        [344.064, 243.971, 337.749, 256.941, 327.187, 428.114, 68.919, 155.867, 270.048],
                        ),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=False))
# training and testing settings
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    use_giou=False,
    use_focal=False,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# dataset settings
dataset_type = 'CustomCocoDataset'
data_root = '../../data/airport'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255, 255, 255], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(input_size, input_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(input_size, input_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=('vehicle', 'person', 'non-vehicle'),
        ann_file=data_root+'/annotation_example_train.json',
        img_prefix=data_root + '/train',
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        classes=('vehicle', 'person', 'non-vehicle'),
        ann_file=data_root+'/annotation_example_val.json',
        img_prefix=data_root + '/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=('vehicle', 'person', 'non-vehicle'),
        ann_file=data_root+'/annotation_example_val.json',
        img_prefix=data_root + '/val',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1.0 / 3,
    step=[50, 75])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 5
# device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'outputs/person-vehicle-bike-detection-crossroad-1016'
load_from = None
resume_from = None
workflow = [('train', 1)]
