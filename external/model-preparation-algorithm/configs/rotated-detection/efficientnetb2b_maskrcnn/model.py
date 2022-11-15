_base_ = [
    "../../../submodule/samples/cfgs/models/backbones/efficientnet_b2b.yaml",
    "../../../submodule/recipes/stages/_base_/models/detectors/efficientnetb2b_maskrcnn.custom.py",
]
evaluation = dict(interval=1, metric="mAP", save_best="mAP", iou_thr=[0.5])
fp16 = dict(loss_scale=512.0)
