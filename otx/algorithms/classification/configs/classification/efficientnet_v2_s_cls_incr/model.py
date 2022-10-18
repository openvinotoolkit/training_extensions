_base_ = [
    "../../../../../../external/model-preparation-algorithm/submodule/models/classification/ote_efficientnet_v2_s.yaml",
]

fp16 = dict(loss_scale=512.0)
