_base_ = ["./datasets/pipelines/rcrop_hflip_resize.py"]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

__dataset_type = "TVDatasetSplit"
__dataset_base = "CIFAR10"

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    num_classes=10,
    pipeline_options=dict(
        Resize=dict(size=224),
        RandomCrop=dict(size=32, padding=4),
        Normalize=dict(mean=[125.307, 122.961, 113.8575], std=[51.5865, 50.847, 51.255]),
        RandomResizedCrop=dict(size=(112, 112)),  # for BYOL pipeline
    ),
    train=dict(
        type=__dataset_type,
        data_prefix="data/torchvision/cifar10",
        base=__dataset_base,
        train=True,
        pipeline=__train_pipeline,
        num_images=5000,
    ),
    val=dict(
        type=__dataset_type,
        base=__dataset_base,
        train=False,
        data_prefix="data/torchvision/cifar10",
        pipeline=__test_pipeline,
        num_images=1000,
    ),
    test=dict(
        type=__dataset_type,
        base=__dataset_base,
        train=False,
        data_prefix="data/torchvision/cifar10",
        pipeline=__test_pipeline,
        num_images=1000,
    ),
    unlabeled=dict(
        type=__dataset_type,
        base=__dataset_base,
        train=True,
        data_prefix="data/torchvision/cifar10",
        # num_images=1000
    ),
)
