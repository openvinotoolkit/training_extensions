"""EfficientNet-B0 config for hierachical classification with contrastive loss for small datasets."""

# pylint: disable=invalid-name

_base_ = "../../base/models/efficientnet.py"

model = dict(
    task="classification",
    type="SupConClassifier",
    backbone=dict(
        version="b0",
    ),
    head=dict(
        type="SupConHierarchicalClsHead",
        in_channels=-1,
        aux_mlp=dict(hid_channels=0, out_channels=1024),
        loss=dict(type="CrossEntropyLoss", reduction="mean", loss_weight=1.0),
        multilabel_loss=dict(
            type="AsymmetricLossWithIgnore",
            gamma_pos=0.0,
            gamma_neg=4.0,
        ),
        aux_loss=dict(
            type="BarlowTwinsLoss",
            off_diag_penality=1.0 / 128.0,
            loss_weight=1.0,
        ),
    ),
)

fp16 = dict(loss_scale=512.0)
