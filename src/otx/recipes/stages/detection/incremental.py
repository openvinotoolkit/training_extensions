_base_ = ["./train.py", "../_base_/models/detectors/detector.py"]

task_adapt = dict(
    type="default_task_adapt",
    op="REPLACE",
    efficient_mode=False,
    use_adaptive_anchor=True,
)

runner = dict(max_epochs=30)

evaluation = dict(interval=1, metric="mAP", save_best="mAP")

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
    dict(
        type="EMAHook",
        priority="ABOVE_NORMAL",
        momentum=0.1,
    ),
]

lr_config = dict(
    policy="ReduceLROnPlateau",
    metric="mAP",
    patience=5,
    iteration_patience=0,
    interval=1,
    min_lr=1e-06,
    warmup="linear",
    warmup_iters=200,
    warmup_ratio=0.3333333333333333,
)

ignore = True
adaptive_validation_interval = dict(
    max_interval=5,
    enable_adaptive_interval_hook=True,
    enable_eval_before_run=True,
)
