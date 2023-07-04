"""EfficientNet-B0 for hierarchical config."""

# pylint: disable=invalid-name

_base_ = ["../../../../recipes/stages/classification/incremental.yaml", "../base/models/efficientnet.py"]

model = dict(
    type="SAMImageClassifier",
    task="classification",
    backbone=dict(
        version="b0",
    ),
    head=dict(
        type="CustomHierarchicalLinearClsHead",
        multilabel_loss=dict(
            type="AsymmetricLossWithIgnore",
            gamma_pos=0.0,
            gamma_neg=4.0,
        ),
    ),
)
