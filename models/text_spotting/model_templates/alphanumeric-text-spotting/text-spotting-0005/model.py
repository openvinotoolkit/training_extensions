# model settings
det_thr = 0.65
rec_thr = 0.45
model = dict(
    type='MaskTextSpotter',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=64,
        feat_channels=64,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHeadWithText',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=64,
            fc_out_channels=512,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=64,
            conv_out_channels=64,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        text_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=28, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        text_head=dict(
            type='TextRecognitionHeadAttention',
            input_feature_size=[28, 28],
            encoder_dim_input=64,
            encoder_dim_internal=256,
            encoder_num_layers=3,
            decoder_input_feature_size=[28, 28],
            decoder_max_seq_len=28,
            decoder_vocab_size=38,
            decoder_dim_hidden=256,
            decoder_sos_index=0,
            decoder_rnn_type='GRU',
            dropout_ratio=0.5
        ),
        text_thr=rec_thr),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            text_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.99,
                neg_iou_thr=0.5,
                min_pos_iou=0.99,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            text_sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=1.0,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            area_per_symbol_thr = 25,
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=300,
            nms_post=300,
            max_num=300,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=det_thr,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5),
        score_thr=det_thr))

dataset_type = 'CocoWithTextDataset'
data_root = 'data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='RandomRotate',
        p=0.5,
        border_mode=0),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_text=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes',
            'gt_texts': 'texts'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Resize',
        img_scale=[(1280, 768), (1280, 768 + 64), (1280, 768 - 64), (1280 + 64, 768), (1280 - 64, 768),
        	    (1216, 704), (1216 - 64, 704), (1216, 704 - 64), (1216 - 64, 704 - 64)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_texts']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            ann_file=[data_root + 'dataset_oiv5_train125f.json',
                      data_root + 'dataset_train_wo_tests_ic13_ic15_tt.json'],
            img_prefix=[data_root, data_root],
            classes=('text', ),
            min_size=0,
            max_texts_num=150,
            pipeline=train_pipeline)
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'icdar2015_test.json',
        img_prefix=data_root,
        classes=('text',),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'icdar2015_test.json',
        img_prefix=data_root,
        classes=('text',),
        pipeline=test_pipeline))

evaluation = dict(metric=['bbox', 'segm', 'f1', 'word_spotting'])
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr=0.00001)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(interval=1)
# yapf:disable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
total_epochs = 25
device_ids = range(4)
lexicon_mapping = 'lexicons/ic15/GenericVocabulary_pair_list.txt'
lexicon = 'lexicons/ic15/GenericVocabulary_new.txt'
