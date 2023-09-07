_base_ = ["./train.py", "../_base_/data/coco_ubt.py", "../_base_/models/detectors/detector.py"]

task_adapt = dict(
    type="default_task_adapt",
    op="REPLACE",
    efficient_mode=False,
    use_adaptive_anchor=True,
)

custom_hooks = [
    dict(
        type="MeanTeacherHook",
        epoch_momentum=0.1,
        start_epoch=2,
    ),
    dict(
        type="LazyEarlyStoppingHook",
        start=3,
        patience=5,
        iteration_patience=1000,
        metric="bbox_mAP",
        interval=1,
        priority=75,
    ),
]

find_unused_parameters = True
