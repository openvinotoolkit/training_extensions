_base_ = [
  '../../../submodule/samples/cfgs/models/backbones/resnet50.yaml',
  '../../../submodule/recipes/stages/_base_/models/detectors/resnet50_maskrcnn.custom.py'
]
fp16 = dict(loss_scale=512.)
