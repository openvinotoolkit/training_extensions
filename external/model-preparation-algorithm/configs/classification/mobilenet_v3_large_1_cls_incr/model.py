_base_ = [
  '../../../submodule/models/classification/ote_mobilenet_v3_large.yaml',
]
fp16 = dict(loss_scale=512.)
