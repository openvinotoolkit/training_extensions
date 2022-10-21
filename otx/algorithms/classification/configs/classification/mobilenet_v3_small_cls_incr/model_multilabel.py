"""MobileNet-V3-Small for multi-label config."""

_base_ = "../../base/models/mobilenet_v3.py"

model = dict(
    type="SAMImageClassifier",
    task="classification",
    head=dict(
        type="CustomMultiLabelNonLinearClsHead",
        loss=dict(
            type="AsymmetricLossWithIgnore",
            gamma_pos=0.0,
            gamma_neg=0.0,
        ),
    ),
)
