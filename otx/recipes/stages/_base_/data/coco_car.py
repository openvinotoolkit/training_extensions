_base_ = [
    './coco.py'
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(classes=('car',)),
    val=dict(classes=('car',)),
    test=dict(classes=('car',))
)
