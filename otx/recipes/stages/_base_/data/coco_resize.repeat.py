_base_ = [
    './data.py',
    './pipelines/coco_pmd_ioucrop_resize.py'
]

__dataset_type = 'CocoDataset'
__data_root = 'data/coco/'

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

__samples_per_gpu = 32

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=2,
    pipeline_options=dict(
        Resize=dict(
            img_scale=(384, 384)
        ),
        MultiScaleFlipAug=dict(
            img_scale=(384, 384)
        )
    ),
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=__dataset_type,
            ann_file=__data_root + 'annotations/instances_train2017.json',
            img_prefix=__data_root + 'train2017/',
            pipeline=__train_pipeline,
            min_size=20
        )
    ),
    val=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_val2017.json',
        img_prefix=__data_root + 'val2017/',
        test_mode=True,
        pipeline=__test_pipeline),
    test=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_val2017.json',
        img_prefix=__data_root + 'val2017/',
        test_mode=True,
        pipeline=__test_pipeline)
)
