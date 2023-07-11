"""Base deit config."""

# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(type="mmcls.VisionTransformer", arch="deit-small", img_size=224, patch_size=16),
    neck=None,
    head=dict(
        type="CustomVisionTransformerClsHead",
        num_classes=1000,
        in_channels=384,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
    init_cfg=[
        dict(type="TruncNormal", layer="Linear", std=0.02),
        dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
    ],
)
