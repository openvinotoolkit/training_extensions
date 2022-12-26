_base_ = [
    './ssd.ote.py'
]

model = dict(
    type='CustomSingleStageDetector',
    bbox_head=dict(type='CustomSSDHead',),
)

ignore = False
