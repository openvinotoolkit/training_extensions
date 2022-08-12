_base_ = [
  '../../../submodule/recipes/stages/_base_/models/detectors/yolox.custom.py'
]
fp16 = dict(loss_scale=512.)
