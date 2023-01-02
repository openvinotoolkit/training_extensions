_base_ = "./model.py"

model = dict(
    type="SupConClassifier",
    task="classification",
    pretrained=None,
    backbone=dict(),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="SupConClsHead",
        num_classes=10,
        in_channels=-1,
        aux_mlp=dict(hid_channels=0, out_channels=1024),
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        aux_loss=dict(
            type="BarlowTwinsLoss",
            off_diag_penality=1.0 / 128.0,
            loss_weight=1.0,
        ),
    ),
)

checkpoint_config = dict(type="CheckpointHookWithValResults")
