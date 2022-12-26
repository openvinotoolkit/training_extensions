_base_ = [
    './datasets/pipelines/rcrop_hflip.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

__dataset_type = 'TVDatasetSplit'
__dataset_base = 'CIFAR10'

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    num_classes=10,
    pipeline_options=dict(
        RandomCrop=dict(size=32, padding=4),
        Normalize=dict(
            mean=[125.307, 122.961, 113.8575],
            std=[51.5865, 50.847, 51.255]
        )
    ),
    train=dict(
        type=__dataset_type,
        data_prefix='data/torchvision/cifar10',
        base=__dataset_base,
        pipeline=__train_pipeline),
    val=dict(
        type=__dataset_type,
        base=__dataset_base,
        data_prefix='data/torchvision/cifar10',
        pipeline=__test_pipeline),
    test=dict(
        type=__dataset_type,
        base=__dataset_base,
        data_prefix='data/torchvision/cifar10',
        pipeline=__test_pipeline)
)
