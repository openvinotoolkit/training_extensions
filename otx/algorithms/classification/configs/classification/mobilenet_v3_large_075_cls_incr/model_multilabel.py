_base_ = "../../base/models/mobilenet_v3.py"

model = dict(
    type="SAMImageClassifier",
    task="classification",
    backbone=dict(
        mode="large",
        width_mult=0.75,
    ),
    head=dict(
        type="CustomMultiLabelNonLinearClsHead",
        in_channels=720,
        hid_channels=1280,
        loss=dict(
            type="AsymmetricLossWithIgnore",
            gamma_pos=0.0,
            gamma_neg=0.0,
        ),
    ),
)
