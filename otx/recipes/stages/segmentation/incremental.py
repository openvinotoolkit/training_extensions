_base_ = ["../_base_/models/segmentors/seg_class_incr.py", "../_base_/data/custom_seg.py"]

optimizer = dict(type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-4, paramwise_cfg={'bias_decay_mult ': 0.0, 'norm_decay_mult ': 0.0})
optimizer_config = dict()
# optimizer_config = dict(
#     _delete_=True,
#     grad_clip=dict(
#         # method='adaptive',
#         # clip=0.2,
#         # method='default',
#         max_norm=40,
#         norm_type=2,
#     ),
# )

lr_config = dict(policy='poly',  warmup='linear',  warmup_iters=260, warmup_ratio=1e-6, power=0.9,  min_lr=1e-6, by_epoch=False)

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

find_unused_parameters = False

task_adapt = dict(
    type="mpa",
    op="REPLACE",
)

ignore = True

cudnn_benchmark = False

deterministic = False

hparams = dict(dummy=0)

