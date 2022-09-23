_base_ = [
    "../../../submodule/samples/cfgs/models/backbones/efficientnet_b2b.yaml",
    "../../../submodule/recipes/stages/_base_/models/detectors/efficientnetb2b_maskrcnn.custom.py",
]
fp16 = dict(loss_scale=512.0)
