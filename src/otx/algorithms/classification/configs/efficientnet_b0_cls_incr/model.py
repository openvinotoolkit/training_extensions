"""EfficientNet-B0 for multi-class config."""

# pylint: disable=invalid-name

_base_ = ["../../../../recipes/stages/classification/incremental.yaml", "../base/models/efficientnet.py"]

model = dict(
    type="SAMImageClassifier",
    task="classification",
    backbone=dict(
        version="b0",
    ),
    head=dict(
        type="CustomLinearClsHead",
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
        ),
    ),
)

fp16 = dict(loss_scale=512.0)
