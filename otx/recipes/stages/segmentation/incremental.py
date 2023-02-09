_base_ = ["./train.py", "../_base_/models/segmentors/seg_class_incr.py", "../_base_/data/custom_seg.py"]

optimizer = dict(_delete_=True, type="Adam", lr=1e-3, eps=1e-08, weight_decay=0.0)

optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(
        # method='adaptive',
        # clip=0.2,
        # method='default',
        max_norm=40,
        norm_type=2,
    ),
)

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True, ignore_last=False),
        # dict(type='TensorboardLoggerHook')
    ],
)

dist_params = dict(backend="nccl", linear_scale_lr=False)

runner = dict(type="EpochRunnerWithCancel", max_epochs=300)

checkpoint_config = dict(by_epoch=True, interval=1)

evaluation = dict(interval=1, metric=["mDice", "mIoU"], show_log=True)

seed = 42
find_unused_parameters = False

task_adapt = dict(
    type="mpa",
    op="REPLACE",
)

ignore = True
