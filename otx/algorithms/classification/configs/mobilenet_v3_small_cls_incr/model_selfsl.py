# _base_ = "../base/models/mobilenet_v3.py"

model = dict(
    task="classification",
    type='SelfSLClassifier',
    backbone=dict(type="OTXMobileNetV3", pretrained=True, mode="small", width_mult=1.0),
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
