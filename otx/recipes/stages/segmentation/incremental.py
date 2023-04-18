_base_ = ["./train.py", "../_base_/models/segmentors/segmentor.py", "../_base_/data/custom_seg.py"]

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0005,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    paramwise_cfg={"bias_decay_mult ": 0.0, "norm_decay_mult ": 0.0},
)
optimizer_config = dict()
# learning policy
lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=300,
    warmup_ratio=1e-6,
    power=0.9,
    min_lr=1e-6,
    by_epoch=False,
)

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True, ignore_last=False),
    ],
)

dist_params = dict(backend="nccl", linear_scale_lr=False)

runner = dict(type="EpochRunnerWithCancel", max_epochs=100)

checkpoint_config = dict(by_epoch=True, interval=1)

evaluation = dict(interval=1, metric=["mDice", "mIoU"], show_log=True)

seed = 42
find_unused_parameters = False

task_adapt = dict(
    type="mpa",
    op="REPLACE",
)

ignore = True
