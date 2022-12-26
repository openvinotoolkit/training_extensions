_base_ = [
    './datasets/types/tv_dataset_split.py',
    './datasets/pipelines/semisl_pipeline.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__train_pipeline_strong = {{_base_.train_pipeline_strong}}
__test_pipeline = {{_base_.test_pipeline}}

__dataset_type = {{_base_.dataset_type}}
__dataset_base = 'STL10'


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    num_classes=10,
    pipeline_options=dict(
        Resize=dict(size=224),
        Normalize=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375]
        )
    ),
    train=[
        # labeled
        dict(
            type=__dataset_type,
            base=__dataset_base,
            data_prefix='data/torchvision/stl10',
            split='train',
            num_images=1000,
            pipeline=__train_pipeline,
            samples_per_gpu=8,
            workers_per_gpu=4,
            download=True,
        ),
        # unlabeled
        dict(
            type=__dataset_type,
            base=__dataset_base,
            split='unlabeled',
            data_prefix='data/torchvision/stl10',
            num_images=7000,
            pipeline=dict(
                weak=__train_pipeline,
                strong=__train_pipeline_strong
            ),
            samples_per_gpu=56,
            workers_per_gpu=4,
            download=True,
            use_labels=False
        )
    ],
    val=dict(
        type=__dataset_type,
        base=__dataset_base,
        split='test',
        data_prefix='data/torchvision/stl10',
        num_images=-1,
        pipeline=__test_pipeline,
        download=True,
    ),
    test=dict(
        type=__dataset_type,
        base=__dataset_base,
        split='test',
        data_prefix='data/torchvision/stl10',
        num_images=-1,
        pipeline=__test_pipeline,
        download=True,
    )
)
