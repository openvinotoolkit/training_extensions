"""Base MobileNet-V3."""

# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(type="otx.OTXMobileNetV3", pretrained=True, mode="small", width_mult=1.0),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="NonLinearClsHead",
        num_classes=1000,
        in_channels=576,
        hid_channels=1024,
        act_cfg=dict(type="HSwish"),
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
