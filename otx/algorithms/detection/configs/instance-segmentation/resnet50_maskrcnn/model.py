_base_ = [
    "../../../../common/adapters/mmcv/configs/backbones/resnet50.yaml",
    "../../base/models/resnet50_maskrcnn.py",
]

model = dict(
    type='CustomMaskRCNN',
    roi_head=dict(
        type='CustomRoIHead',
    )
)
