_base_ = [
    "../../../submodule/models/classification/ote_efficientnet_b0_multilabel_al.yaml",
]

fp16 = dict(loss_scale=512.0)
