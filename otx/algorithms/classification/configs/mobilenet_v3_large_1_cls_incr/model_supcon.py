"""MobileNet-V3-large-1 for multi-class config."""

# pylint: disable=invalid-name

_base_ = "../base/models/mobilenet_v3.py"

model = dict(
    type="SupConClassifier",
    task="classification",
    backbone=dict(mode="large"),
    head=dict(
        type="SupConClsHead",
        in_channels=-1,
        aux_mlp=dict(hid_channels=0, out_channels=1024),
        loss=dict(
            type="BarlowTwinsLoss",
            off_diag_penality=1.0 / 128.0,
            loss_weight=1.0,
            cls_loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        ),
    ),
)

fp16 = dict(loss_scale=512.0)
