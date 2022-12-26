_base_ = [
    './model.py'
]

model = dict(
    type='SelfSL',
    pretrained=None,
    base_momentum=0.99,
    backbone=dict(
        cfg=dict(type=''),
        reg=None
    ),
    neck=dict(
        type='MLP',
        in_channels=-1,
        hid_channels=4096,
        out_channels=256,
        norm_cfg=dict(type='BN2d'),
        use_conv=True,
        with_avg_pool=False
    ),
    head=dict(
        type='LatentPredictHead',
        loss='PPC',
        predictor=dict(
            type='PPM',
            sharpness=2
        ),
        pos_ratio=0.7
    )
)

custom_hooks = [
    dict(
        type='SelfSLHook',
        end_momentum=1.
    )
]
