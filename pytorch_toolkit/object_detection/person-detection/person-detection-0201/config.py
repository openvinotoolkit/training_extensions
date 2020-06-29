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
                [image_width * x for x in [0.0221902727105487, 0.054625211024679, 0.102332663312817, 0.14364042118466025]],
                [image_width * x for x in [0.20817454792318996, 0.45909577824808057, 0.2758748647618599, 0.5238139227422672, 0.8531110786213814]],               
            ],
            heights=[
	        [image_height * x for x in [0.06031581928255733, 0.14855637858557702, 0.2668119832636703, 0.4203179599319901]],
		[image_height * x for x in [0.6082611972029469, 0.44005493324397005, 0.8207143765730922, 0.8343620047507052, 0.851994025022708]],
	    ],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2),),
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
data_root = '/media/cluster_fs/user/yurygoru/crossroad_extra/'
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
    dict(type='RandomFlip', flip_ratio=0.0),
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
    samples_per_gpu=65,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            classes=('person',),
            ann_file='/media/cluster_fs/datasets/crossroad_extra_2.0_1920x1080_limited/annotation_train/instances_person_train.json',
            min_size=20,
            img_prefix='/media/cluster_fs/datasets/crossroad_extra_2.0_1920x1080_limited/',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        classes=('person',),
        ann_file='/media/cluster_fs/datasets/crossroad_extra_2.0/annotation_trainval/instances_person_trainval.json',
        img_prefix='/media/cluster_fs/datasets/crossroad_extra_2.0',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=('person',),
        ann_file='/media/cluster_fs/datasets/crossroad_extra_2.0/annotation_trainval/instances_person_trainval.json',
        img_prefix='/media/cluster_fs/datasets/crossroad_extra_2.0',
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
    step=[8, 11, 13])
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
total_epochs = 14
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/media/cluster_fs/user/ikrylov/experiments/person_detection/0201_cr2.0/outputs'
load_from = None
resume_from = None
workflow = [('train', 1)]
