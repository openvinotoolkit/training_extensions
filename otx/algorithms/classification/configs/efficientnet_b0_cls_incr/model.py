"""EfficientNet-B0 for multi-class config."""

# pylint: disable=invalid-name

#_base_ = "../base/models/efficientnet.py"

model = dict(
    backbone=dict(type="OTXEfficientNet", pretrained=True, version="b0"),
    neck=dict(type="GlobalAveragePooling"),
)

# cls should use SemiClassifier for semi-sl
"""
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
"""
fp16 = dict(loss_scale=512.0)
