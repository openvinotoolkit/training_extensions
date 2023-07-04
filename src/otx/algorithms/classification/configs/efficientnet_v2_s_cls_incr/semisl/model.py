"""EfficientNet-V2 config for semi-supervised multi-class classification."""

# pylint: disable=invalid-name

_base_ = ["../../../../../recipes/stages/classification/semisl.yaml", "../../base/models/efficientnet_v2.py"]

model = dict(
    type="SemiSLClassifier",
    task="classification",
    backbone=dict(
        version="s_21k",
    ),
    head=dict(type="SemiLinearClsHead", loss=dict(type="CrossEntropyLoss", loss_weight=1.0)),
)

fp16 = dict(loss_scale=512.0)
