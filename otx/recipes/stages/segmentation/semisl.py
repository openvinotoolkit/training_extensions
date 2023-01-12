_base_ = [
    "../_base_/data/seg_semisl.py",
    "../_base_/models/segmentors/segmentor.py",
    "./train.py",
]

optimizer = dict(_delete_=True, type="Adam", lr=1e-3, eps=1e-08, weight_decay=0.0)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=40, norm_type=2))

custom_hooks = [
    dict(
        type="DualModelEMAHook",
        momentum=0.99,
        start_epoch=1,
        src_model_name="model_s",
        dst_model_name="model_t",
    ),
]

log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True),
    ],
)

dist_params = dict(backend="nccl", linear_scale_lr=False)

runner = dict(type="EpochRunnerWithCancel", max_epochs=300)

checkpoint_config = dict(
    by_epoch=True,
    interval=1,
)

evaluation = dict(interval=1, metric=["mDice", "mIoU"], show_log=True)

task_adapt = dict(
    op="REPLACE",
)
ignore = True


find_unused_parameters = True
seed = 42
