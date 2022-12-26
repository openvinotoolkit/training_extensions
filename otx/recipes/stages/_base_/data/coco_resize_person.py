_base_ = [
    './coco_resize.py'
]

data = dict(
    train=dict(classes=('person',)),
    val=dict(classes=('person',)),
    test=dict(classes=('person',))
)
