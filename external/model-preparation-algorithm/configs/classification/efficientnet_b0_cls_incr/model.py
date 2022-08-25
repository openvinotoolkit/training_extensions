_base_ = [
  '../../../submodule/models/classification/ote_efficientnet_b0.yaml',
]

runner = dict(max_epochs=20)
fp16 = dict(loss_scale=512.)
