"""MobileNet-V3-large-1 for multi-label config."""

# pylint: disable=invalid-name

_base_ = ["../../../../recipes/stages/classification/multilabel/incremental.yaml", "../base/models/mobilenet_v3.py"]

model = dict(
    type="SAMImageClassifier",
    task="classification",
    backbone=dict(mode="large"),
    head=dict(
        type="CustomMultiLabelNonLinearClsHead",
        in_channels=960,
        hid_channels=1280,
        normalized=True,
        scale=7.0,
        act_cfg=dict(
            type="PReLU",
        ),
        loss=dict(
            type="AsymmetricAngularLossWithIgnore",
            gamma_pos=0.0,
            gamma_neg=1.0,
            reduction="sum",
        ),
    ),
)

fp16 = dict(loss_scale=512.0)
