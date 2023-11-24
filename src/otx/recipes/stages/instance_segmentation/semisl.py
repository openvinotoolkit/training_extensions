_base_ = ["./train.py", "../_base_/models/detectors/detector.py"]

task = "instance-segmentation"

task_adapt = dict(
    type="default_task_adapt",
    op="REPLACE",
    efficient_mode=False,
)

runner = dict(max_epochs=300)

optimizer_config = dict(_delete_=True, grad_clip=None)

ignore = True
find_unused_parameters = True

adaptive_validation_interval = dict(
    max_interval=5,
    enable_adaptive_interval_hook=True,
    enable_eval_before_run=True,
)

custom_hooks = [
    dict(type="MeanTeacherHook", epoch_momentum=0.1, start_epoch=5),
]
