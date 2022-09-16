_base_ = [
    "../../../submodule/models/classification/ote_mobilenet_v3_large.yaml",
]

runner = dict(max_epochs=20)
fp16 = dict(loss_scale=512.0)
