"""MobileNet-V3-large-075 config for multi-label classification with contrastive loss for small datasets."""

# pylint: disable=invalid-name

_base_ = "../../base/models/mobilenet_v3.py"

model = dict(
    task="classification",
    type="SupConClassifier",
    backbone=dict(
        mode="large",
        width_mult=0.75,
    ),
    head=dict(
        type="SupConMultiLabelClsHead",
        in_channels=-1,
        aux_mlp=dict(hid_channels=0, out_channels=1024),
        loss=dict(
            type="AsymmetricLossWithIgnore",
            gamma_pos=0.0,
            gamma_neg=0.0,
        ),
        aux_loss=dict(
            type="BarlowTwinsLoss",
            off_diag_penality=1.0 / 128.0,
            loss_weight=1.0,
        ),
    ),
)
