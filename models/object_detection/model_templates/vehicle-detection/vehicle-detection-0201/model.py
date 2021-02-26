# model settings
input_size = 384
image_width, image_height = input_size, input_size
width_mult = 1.0
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(4, 5),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True
    ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        num_classes=1,
        in_channels=(int(width_mult * 96), int(width_mult * 320)),
        anchor_generator=dict(
            type='SSDAnchorGeneratorClustered',
            strides=(16, 32),
            widths=[
                [image_width * x for x in
                 [0.03190665436176343, 0.07246889234100704, 0.13534887659200423,
                  0.15785373692633903]],
                [image_width * x for x in
                 [0.29641301747833876, 0.3296121909277462, 0.2130097277660658, 0.45965227145427356,
                  0.6790618020935402]],

            ],
            heights=[
                [image_height * x for x in
                 [0.03836520511012714, 0.08658721963691468, 0.15106894690367778,
                  0.30385265971706454]],
                [image_height * x for x in
                 [0.2010092421878013, 0.4087630480186569, 0.6694307467997506, 0.4766551262641776,
                  0.6573119829938249]],
            ],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2), ),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=True))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.4,
        neg_iou_thr=0.4,
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
# model training and testing settings
# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
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
        min_crop_size=0.1),
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
    samples_per_gpu=82,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            classes=('vehicle',),
            ann_file='data/annotations/instances_train2017car.json',
            img_prefix='data/train2017',
            min_size=20,
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        classes=('vehicle',),
        ann_file='data/annotations/instances_val2017car.json',
        img_prefix='data/val2017',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=('vehicle',),
        ann_file='data/annotations/instances_val2017car.json',
        img_prefix='data/val2017',
        test_mode=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1.0 / 3,
    step=[8, 15, 18])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'outputs/vehicle-detection-0201'
load_from = None
resume_from = None
workflow = [('train', 1)]
