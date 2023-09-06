_base_ = ["./train.py", "../_base_/data/coco_ubt.py", "../_base_/models/detectors/detector.py"]

model = dict(super_type="UnbiasedTeacher")  # Used as general framework

custom_hooks = [
    dict(
        type="DualModelEMAHook",
        epoch_momentum=0.4,
        start_epoch=2,
    ),
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
