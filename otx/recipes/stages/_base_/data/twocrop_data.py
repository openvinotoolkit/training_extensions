_base_ = ["./pipelines/twocrop_pipeline.py"]

__dataset_type = "OTXClsDataset"

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

__samples_per_gpu = 16

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=2,
    train=dict(type=__dataset_type, pipeline=__train_pipeline),
    val=dict(type=__dataset_type, test_mode=True, pipeline=__test_pipeline),
    test=dict(type=__dataset_type, test_mode=True, pipeline=__test_pipeline),
)
