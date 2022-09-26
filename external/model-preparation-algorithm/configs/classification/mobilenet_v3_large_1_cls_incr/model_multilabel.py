_base_ = [
    "../../../submodule/models/classification/ote_mobilenet_v3_large_multilabel.yaml",
]

fp16 = dict(loss_scale=512.0)
