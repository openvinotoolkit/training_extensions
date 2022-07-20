_base_ = [
    './data_pipeline.py'
]

# pre-trained params settings
ignore_keys = [r'^backbone\.increase_modules\.', r'^backbone\.increase_modules\.',
               r'^backbone\.downsample_modules\.', r'^backbone\.final_layer\.',
               r'^head\.']

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='LiteHRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stem=dict(
                stem_channels=32,
                out_channels=32,
                expand_ratio=1,
                strides=(2, 2),
                extra_stride=True,
                input_norm=False,
            ),
            num_stages=2,
            stages_spec=dict(
                neighbour_weighting=False,
                weighting_module_version='v1',
                num_modules=(4, 4),
                num_branches=(2, 3),
                num_blocks=(2, 2),
                module_type=('LITE', 'LITE'),
                with_fuse=(True, True),
                reduce_ratios=(8, 8),
                num_channels=(
                    (60, 120),
                    (60, 120, 240),
                )
            ),
            out_modules=dict(
                conv=dict(
                    enable=False,
                    channels=160
                ),
                position_att=dict(
                    enable=False,
                    key_channels=64,
                    value_channels=240,
                    psp_size=(1, 3, 6, 8),
                ),
                local_att=dict(
                    enable=False
                )
            ),
            out_aggregator=dict(
                enable=False
            ),
            add_input=False
        )
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=[60, 120, 240],
        in_index=[0, 1, 2],
        input_transform='multiple_select',
        channels=60,
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        enable_aggregator=True,
        aggregator_merge_norm=None,
        aggregator_use_concat=False,
        enable_out_norm=False,
        enable_loss_equalizer=True,
        loss_decode=[
            dict(type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_jitter_prob=0.01,
                sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                loss_weight=4.0),
            dict(type='GeneralizedDiceLoss',
                smooth=1.0,
                gamma=5.0,
                alpha=0.5,
                beta=0.5,
                focal_gamma=1.0,
                loss_jitter_prob=0.01,
                loss_weight=4.0),
        ]
    ),
    train_cfg=dict(
        mix_loss=dict(
            enable=False,
            weight=0.1
        ),
    ),
    test_cfg=dict(
        mode='whole',
        output_scale=10.0,
    ),
)

find_unused_parameters = False

# optimizer
optimizer = dict(
    type='Adam',
    lr=1e-3,
    eps=1e-08,
    weight_decay=0.0
)
optimizer_config = dict(
    grad_clip=dict(
        method='default',
        max_norm=40,
        norm_type=2
    )
)

# parameter manager
params_config = dict(
    type='FreezeLayers',
    by_epoch=True,
    iters=0,
    open_layers=[r'backbone\.aggregator\.', r'neck\.', r'decode_head\.', r'auxiliary_head\.']
)

# learning policy
lr_config = dict(
    policy='customstep',
    gamma=0.1,
    by_epoch=True,
    step=[400, 500],
    fixed='constant',
    fixed_iters=0,
    fixed_ratio=10.0,
    warmup='cos',
    warmup_iters=80,
    warmup_ratio=1e-2,
)

# runtime settings
runner = dict(
    type='EpochBasedRunner',
    max_epochs=600
)
checkpoint_config = dict(
    by_epoch=True,
    interval=1
)
evaluation = dict(
    by_epoch=True,
    interval=1,
    metric='mDice'
)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnetsv2_imagenet1k_rsc.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
