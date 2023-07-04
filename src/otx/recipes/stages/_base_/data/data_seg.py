_base_ = ["./pipelines/incr_seg.py"]

__dataset_type = ""
__data_root = ""
__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=1,
        dataset=dict(
            type=__dataset_type,
            data_root=__data_root,
            img_dir=None,
            ann_dir=None,
            split=None,
            pipeline=__train_pipeline,
        ),
    ),
    val=dict(
        type=__dataset_type, data_root=__data_root, img_dir=None, ann_dir=None, split=None, pipeline=__test_pipeline
    ),
    test=dict(
        type=__dataset_type, data_root=__data_root, img_dir=None, ann_dir=None, split=None, pipeline=__test_pipeline
    ),
)
