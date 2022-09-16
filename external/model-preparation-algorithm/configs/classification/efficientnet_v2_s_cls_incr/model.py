_base_ = [
  '../../../submodule/models/classification/ote_efficientnet_v2_s.yaml',
]

runner = dict(max_epochs=20)
fp16 = dict(loss_scale=512.)
