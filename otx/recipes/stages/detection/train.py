_base_ = [
    "../_base_/default.py",
    "../_base_/data/data.py",
    "../_base_/logs/tensorboard_logger.py",
    "../_base_/optimizers/sgd.py",
    "../_base_/runners/epoch_runner_cancel.py",
    "../_base_/schedules/plateau.py",
]

# Disabling until evaluation bug resolved
# data = dict(
#     val=dict(samples_per_gpu=2),
#     test=dict(samples_per_gpu=2),
# )

optimizer = dict(
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
)

optimizer_config = dict(
    _delete_=True,
    type="OptimizerHook",
    #  type="SAMOptimizerHook",
    grad_clip=dict(max_norm=35, norm_type=2),
)

lr_config = dict(min_lr=1e-06)

evaluation = dict(interval=1, metric="bbox", classwise=True, save_best="bbox_mAP")

custom_hooks = [
    dict(
        type="LazyEarlyStoppingHook",
        start=3,
        patience=10,
        iteration_patience=0,
        metric="bbox_mAP",
        interval=1,
        priority=75,
    ),
]
