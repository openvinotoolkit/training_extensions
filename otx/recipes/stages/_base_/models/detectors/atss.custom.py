_base_ = [
    './atss.py'
]

model = dict(
    type='CustomATSS',
    bbox_head=dict(
        type='CustomATSSHead',
        use_qfl=False,
        qfl_cfg=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
    ),
)
