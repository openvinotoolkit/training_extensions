_base_ = [
    "../../../submodule/samples/cfgs/models/backbones/resnet50.yaml",
    "../../../submodule/recipes/stages/_base_/models/detectors/resnet50_maskrcnn.custom.py",
]
evaluation = dict(interval=1, metric="mAP", save_best="mAP", iou_thr=[0.5])
