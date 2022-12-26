_base_ = [
    './model.py'
]

model = dict(
    type='SelfSL',
    pretrained=None,
    base_momentum=0.996,
    backbone=dict(
        cfg=dict(type=''),
        reg=None
    ),
    neck=dict(
        type='MLP',
        in_channels=-1,
        hid_channels=4096,
        out_channels=256,
        with_avg_pool=True
    ),
    head=dict(
        type='LatentPredictHead',
        loss='MSE',
        predictor=dict(
            type='MLP',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            with_avg_pool=False
        )
    )
)

custom_hooks = [
    dict(
        type='SelfSLHook',
        end_momentum=1.
    )
]