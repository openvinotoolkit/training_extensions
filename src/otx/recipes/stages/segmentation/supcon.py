_base_ = ["./incremental.py"]

custom_hooks = [
    dict(
        type="LazyEarlyStoppingHook",
        patience=8,
        iteration_patience=0,
        metric="mDice",
        interval=1,
        priority=75,
        start=1,
    ),
    dict(type="TwoCropTransformHook", interval=5),
]

# added to support multi-gpu training
find_unused_parameters = True
