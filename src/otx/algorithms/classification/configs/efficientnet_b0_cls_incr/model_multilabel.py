"""EfficientNet-B0 for multi-label config."""

# pylint: disable=invalid-name

_base_ = ["../../../../recipes/stages/classification/multilabel/incremental.yaml", "../base/models/efficientnet.py"]

model = dict(
    type="SAMImageClassifier",
    task="classification",
    backbone=dict(
        version="b0",
    ),
    head=dict(
        type="CustomMultiLabelLinearClsHead",
        normalized=True,
        scale=7.0,
        loss=dict(type="AsymmetricAngularLossWithIgnore", gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
    ),
)

fp16 = dict(loss_scale=512.0)
