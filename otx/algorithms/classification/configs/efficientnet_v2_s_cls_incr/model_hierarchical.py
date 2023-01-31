"""EfficientNet-V2 for hierarchical config."""

# pylint: disable=invalid-name

_base_ = "../base/models/efficientnet_v2.py"

model = dict(
    type="SAMImageClassifier",
    task="classification",
    backbone=dict(version="s_21k"),
    head=dict(
        type="CustomHierarchicalLinearClsHead",
        multilabel_loss=dict(
            type="AsymmetricLossWithIgnore",
            gamma_pos=0.0,
            gamma_neg=4.0,
        ),
    ),
)
