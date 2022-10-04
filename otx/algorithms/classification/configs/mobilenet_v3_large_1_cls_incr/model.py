_base_ = [
    "../../../../../external/model-preparation-algorithm/submodule/models/classification/ote_mobilenet_v3_large.yaml",
]

fp16 = dict(loss_scale=512.0)
