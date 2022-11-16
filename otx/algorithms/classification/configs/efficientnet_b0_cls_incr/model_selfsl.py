# _base_ = [ '../../configs/base/models/efficientnet.py' ]

model = dict(
    task="classification",
    type='SelfSLClassifier',
    backbone=dict(type="OTXEfficientNet", pretrained=True, version="b0"),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='SelfSLClsHead',
        num_classes=10,
        in_channels=-1,
        aux_head=dict(hid_channels=0, out_channels=1024),
        loss=dict(
            type='BarlowTwinsLoss',
            off_diag_penality=1. / 128.
        )
    )
)

checkpoint_config = dict(
    type='CheckpointHookWithValResults'
)
