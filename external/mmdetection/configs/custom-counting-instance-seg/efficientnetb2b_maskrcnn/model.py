_base_ = [
    './coco_data_pipeline.py'
]

model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='efficientnet_b2b',
        out_indices=(2, 3, 4, 5),
        frozen_stages=-1,
        pretrained=True,
        activation_cfg=dict(type='torch_swish'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=dict(
        type='FPN',
        in_channels=[24, 48, 120, 352],
        out_channels=80,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=80,
        feat_channels=80,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=80,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=80,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=80,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=80,
            conv_out_channels=80,
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                gpu_assign_thr=300),
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
            nms_thr=0.8,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1,
                gpu_assign_thr=300),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=800,
            nms_post=500,
            max_num=500,
            nms_thr=0.8,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=500,
            mask_thr_binary=0.5)))

cudnn_benchmark = True
evaluation = dict(interval=1, metric='mAP', save_best='mAP', iou_thr=[0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9, 0.95])
optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='mAP',
    patience=5,
    iteration_patience=300,
    interval=1,
    min_lr=0.000001,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.3333333333333333
)
runner = dict(type='EpochRunnerWithCancel', max_epochs=300)
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/instance_segmentation/v2/efficientnet_b2b-mask_rcnn-576x576.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = 'output'
custom_hooks = [
    dict(type='EarlyStoppingHook', patience=10, metric='mAP',
         interval=1, priority=75, iteration_patience=0),
]
fp16 = dict(loss_scale=512.)