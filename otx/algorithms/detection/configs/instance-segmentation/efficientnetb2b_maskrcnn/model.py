_base_ = [
    "../../../../common/adapters/mmcv/configs/backbones/efficientnet_b2b.yaml",
    "../../base/models/efficientnetb2b_maskrcnn.py",
]


model = dict(
    type='CustomMaskRCNN',
    roi_head=dict(
        type='CustomRoIHead',
    )
)

fp16 = dict(loss_scale=512.0)
