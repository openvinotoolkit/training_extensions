_base_ = [
    './datasets/pipelines/semisl_pipeline.py'
]

__dataset_type = 'TVDatasetSplit'
__train_pipeline = {{_base_.train_pipeline}}
__train_pipeline_strong = {{_base_.train_pipeline_strong}}

__test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    num_classes=10,
    pipeline_options=dict(
        Resize=dict(
            size=224
        ),
        Normalize=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375]
        )
    ),
    train=[
        # labeled
        dict(
            type=__dataset_type,
            base='CIFAR10',
            train=True,
            data_prefix='data/torchvision/cifar10',
            num_images=1000,
            pipeline=__train_pipeline,
            samples_per_gpu=8,
            workers_per_gpu=4,
            download=True
        ),
        # unlabeled
        dict(
            type=__dataset_type,
            base='CIFAR10',
            train=True,
            data_prefix='data/torchvision/cifar10',
            num_images=7000,
            pipeline=dict(
                weak=__train_pipeline,
                strong=__train_pipeline_strong
            ),
            samples_per_gpu=56,
            workers_per_gpu=4,
            download=True,
            use_labels=False,
        )
    ],
    val=dict(
        type=__dataset_type,
        base='CIFAR10',
        data_prefix='data/torchvision/cifar10',
        num_images=1000,
        pipeline=__test_pipeline,
        download=True
    ),
    test=dict(
        type='CIFAR10',
        data_prefix='data/torchvision/cifar10',
        pipeline=__test_pipeline,
        test_mode=True
    )
)
