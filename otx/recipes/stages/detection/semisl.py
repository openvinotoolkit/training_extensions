_base_ = ["./train.py", "../_base_/data/coco_ubt.py", "../_base_/models/detectors/detector.py"]

task_adapt = dict(
    type="mpa",
    op="REPLACE",
    efficient_mode=False,
)

custom_hooks = [
    dict(
        type="UnbiasedTeacherHook",
        epoch_momentum=0.1,
        start_epoch=2,
        # min_pseudo_label_ratio=0.1,
        min_pseudo_label_ratio=0.0,
    ),
    dict(
        type="DualModelEMAHook",
        epoch_momentum=0.4,
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
