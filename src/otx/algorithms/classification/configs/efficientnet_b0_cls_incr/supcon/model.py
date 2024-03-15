"""EfficientNet-B0 config for multi-class with contrastive loss for small datasets."""

# pylint: disable=invalid-name

_base_ = ["../../../../../recipes/stages/classification/supcon.yaml", "../../base/models/efficientnet.py"]

model = dict(
    task="classification",
    type="SupConClassifier",
    backbone=dict(
        version="b0",
    ),
    head=dict(
        type="SupConClsHead",
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

fp16 = dict(loss_scale=512.0)
