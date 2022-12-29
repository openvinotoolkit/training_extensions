_base_ = [
    './data.py',
]

__dataset_type = 'CocoDataset'
__data_root = 'data/coco/'

__samples_per_gpu = 2

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_train2017.json',
        img_prefix=__data_root + 'train2017/',
    ),
    val=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_val2017.json',
        img_prefix=__data_root + 'val2017/',
        test_mode=True,
    ),
    test=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_val2017.json',
        img_prefix=__data_root + 'val2017/',
        test_mode=True,
    )
)

