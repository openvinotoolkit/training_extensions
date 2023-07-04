_base_ = [
    "../_base_/logs/tensorboard_logger.py",
    "../_base_/optimizers/sgd.py",
    "../_base_/runners/epoch_runner.py",
    "../_base_/data/selfsl_seg_data.py",
]


# doesn't inherit default.py to disenable task_adapt
cudnn_benchmark = True

seed = 5
deterministic = False

hparams = dict(dummy=0)

optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=10e-4)

lr_config = dict(policy="step", by_epoch=True, gamma=1, step=10)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=40, norm_type=2))

log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True, ignore_last=False),
    ],
)

runner = dict(type="EpochBasedRunner", max_epochs=10)

checkpoint_config = dict(by_epoch=True, interval=1)

find_unused_parameters = False
