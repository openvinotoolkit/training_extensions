_base_ = [
    './efficientnetb2b_maskrcnn.py'
]

model = dict(
    type='CustomMaskRCNN',
    roi_head=dict(
        type='CustomRoIHead',
    )
)
