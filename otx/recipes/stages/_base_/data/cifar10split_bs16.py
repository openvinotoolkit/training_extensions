_base_ = [
    './datasets/pipelines/rcrop_hflip.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

__dataset_type = 'TVDatasetSplit'
__dataset_base = 'CIFAR10'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
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
        base=__dataset_base,
        train=True,
        data_prefix='data/torchvision/cifar10',
        num_images=1000,
        pipeline=__train_pipeline,
        download=True
    ),
    val=dict(
        type=__dataset_type,
        base=__dataset_base,
        train=False,
        data_prefix='data/torchvision/cifar10',
        num_images=500,
        pipeline=__test_pipeline,
        download=True
    ),
    test=dict(
        type=__dataset_type,
        base=__dataset_base,
        train=False,
        data_prefix='data/torchvision/cifar10',
        num_images=1000,
        pipeline=__test_pipeline,
        download=True
    )
)
