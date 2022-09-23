_base_ = [
    "../../../submodule/models/classification/ote_efficientnet_v2_s_multilabel.yaml",
]

fp16 = dict(loss_scale=512.0)
