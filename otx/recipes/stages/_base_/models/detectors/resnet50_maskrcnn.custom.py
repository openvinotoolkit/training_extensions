_base_ = [
    './resnet50_maskrcnn.py'
]

model = dict(
    type='CustomMaskRCNN',
    roi_head=dict(
        type='CustomRoIHead',
    )
)
