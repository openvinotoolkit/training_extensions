__dataset_type = ""
__data_root = ""
__pipeline = ""

data = dict(
    train=dict(
        type="RepeatDataset", times=1, dataset=dict(type=__dataset_type, data_root=__data_root, pipeline=__pipeline)
    ),
    val=dict(type=__dataset_type, data_root=__data_root, pipeline=__pipeline),
    test=dict(type=__dataset_type, data_root=__data_root, pipeline=__pipeline),
    unlabeled=dict(type=__dataset_type, data_root=__data_root, pipeline=__pipeline),
)
