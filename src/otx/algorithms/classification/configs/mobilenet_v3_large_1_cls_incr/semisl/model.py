"""MobileNet-V3-large-1 config for semi-supervised multi-class classification."""

# pylint: disable=invalid-name

_base_ = ["../../../../../recipes/stages/classification/semisl.yaml", "../../base/models/mobilenet_v3.py"]

model = dict(
    type="SemiSLClassifier",
    task="classification",
    backbone=dict(mode="large"),
    head=dict(
        type="SemiNonLinearClsHead",
        in_channels=960,
        hid_channels=1280,
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
        ),
    ),
)

fp16 = dict(loss_scale=512.0)
