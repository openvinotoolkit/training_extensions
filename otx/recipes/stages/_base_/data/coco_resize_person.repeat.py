# data settings
_base_ = [
    './coco_resize.repeat.py'
]

data = dict(
    train=dict(
        dataset=dict(
            classes=('person',)
        ),
    ),
    val=dict(
        classes=('person',)
    ),
    test=dict(
        classes=('person',)
    )
)
