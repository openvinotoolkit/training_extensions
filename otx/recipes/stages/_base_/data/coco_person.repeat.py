# flake8: noqa
_base_ = [
    './coco.repeat.py'
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        dataset=dict(classes=('person',))
    ),
    val=dict(classes=('person',)),
    test=dict(classes=('person',))
)
