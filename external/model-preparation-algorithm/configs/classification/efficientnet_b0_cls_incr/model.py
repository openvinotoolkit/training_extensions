_base_ = [
    "../../../submodule/models/classification/ote_efficientnet_b0.yaml",
]

fp16 = dict(loss_scale=512.0)
