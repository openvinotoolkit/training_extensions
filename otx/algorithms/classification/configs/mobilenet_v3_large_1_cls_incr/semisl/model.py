"""MobileNet-V3-large-1 for multi-class config."""

# pylint: disable=invalid-name

_base_ = "../../base/models/mobilenet_v3.py"

model = dict(
    type="SemiSLClassifier",
    task="classification",
    backbone=dict(mode="large"),
    head=dict(
        _delete_=True,
        type="SemiSLClsHead",
        in_channels=960,
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
        ),
    ),
)

fp16 = dict(loss_scale=512.0)
