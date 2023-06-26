_base_ = ["./train.py", "../_base_/data/coco_inst_seg.py", "../_base_/models/detectors/detector.py"]

task = "instance-segmentation"

task_adapt = dict(
    type="mpa",
    op="REPLACE",
    efficient_mode=False,
)

runner = dict(max_epochs=300)

optimizer_config = dict(_delete_=True)

adaptive_validation_interval = dict(
    max_interval=12,
    enable_adaptive_interval_hook=False,
    enable_eval_before_run=True,
)

custom_hooks = [
    dict(
        type="UnbiasedTeacherHook",
        epoch_momentum=0.1,
        start_epoch=8,
        min_pseudo_label_ratio=0.1,
        # min_pseudo_label_ratio=0.0,
    ),
    dict(
        type="DualModelEMAHook",
        epoch_momentum=0.4,
        start_epoch=8,
    )
]

ignore = True
find_unused_parameters = True
