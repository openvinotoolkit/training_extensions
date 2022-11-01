"""MobileNet-V3-Small for hierarchical config."""

# pylint: disable=invalid-name

_base_ = "../base/models/mobilenet_v3.py"

model = dict(
    type="SAMImageClassifier",
    task="classification",
    head=dict(
        type="CustomHierarchicalNonLinearClsHead",
        multilabel_loss=dict(
            type="AsymmetricLossWithIgnore",
            gamma_pos=0.0,
            gamma_neg=4.0,
        ),
    ),
)
