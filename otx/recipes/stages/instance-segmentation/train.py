_base_ = [
    "../_base_/default.py",
    "../_base_/data/data.py",
    "../_base_/logs/tensorboard_logger.py",
    "../_base_/optimizers/sgd.py",
    "../_base_/runners/epoch_runner_cancel.py",
    "../_base_/schedules/plateau.py",
]

optimizer = dict(
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
)

lr_config = dict(
    policy="ReduceLROnPlateau",
    metric="mAP",
    patience=5,
    iteration_patience=0,
    interval=1,
    min_lr=0.000001,
    warmup="linear",
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
)

evaluation = dict(interval=1, metric="mAP", classwise=True, save_best="mAP")

custom_hooks = [
    dict(
        type="LazyEarlyStoppingHook",
        start=3,
        patience=10,
        iteration_patience=0,
        metric="mAP",
        interval=1,
        priority=75,
    ),
]
