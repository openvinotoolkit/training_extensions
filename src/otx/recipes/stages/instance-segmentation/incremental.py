_base_ = ["./train.py", "../_base_/models/detectors/detector.py"]

task = "instance-segmentation"

evaluation = dict(
    interval=1, metric="mAP", save_best="mAP", iou_thr=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
)

task_adapt = dict(
    type="mpa",
    op="REPLACE",
    efficient_mode=False,
)

runner = dict(max_epochs=300)

optimizer_config = dict(_delete_=True)

ignore = True
adaptive_validation_interval = dict(
    max_interval=5,
    enable_adaptive_interval_hook=True,
    enable_eval_before_run=True,
)
