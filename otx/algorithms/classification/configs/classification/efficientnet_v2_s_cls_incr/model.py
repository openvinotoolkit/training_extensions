"""EfficientNet-V2 for multi-class config."""

_base_ = "../../base/models/efficientnet_v2.py"

model = dict(
    type="SAMImageClassifier",
    task="classification",
    backbone=dict(
        version="s_21k",
    ),
    head=dict(type="CustomLinearClsHead", loss=dict(type="CrossEntropyLoss", loss_weight=1.0)),
)

fp16 = dict(loss_scale=512.0)
