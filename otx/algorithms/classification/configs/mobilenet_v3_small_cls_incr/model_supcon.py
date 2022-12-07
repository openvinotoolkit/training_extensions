"""MobileNet-V3-Small for multi-class config."""

# pylint: disable=invalid-name

_base_ = "../base/models/mobilenet_v3.py"

model = dict(
    type="SupConClassifier",
    task="classification",
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
