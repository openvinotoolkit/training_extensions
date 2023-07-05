_base_ = ["./train.py", "../_base_/data/coco_inst_seg.py", "../_base_/models/detectors/detector.py"]

task = "instance-segmentation"

task_adapt = dict(
    type="mpa",
    op="REPLACE",
    efficient_mode=False,
)

runner = dict(max_epochs=300)

optimizer_config = dict(_delete_=True, grad_clip=None)

adaptive_validation_interval = dict(
    max_interval=12,
    enable_adaptive_interval_hook=False,
    enable_eval_before_run=True,
)

custom_hooks = [
    dict(
        type="UnbiasedTeacherHook",
        epoch_momentum=0.0,
        start_epoch=16
    )
]

adaptive_ema = dict(epoch_momentum=0.4)
ignore = True
find_unused_parameters = True
