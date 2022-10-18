_base_ = [
    "../../../../../../external/model-preparation-algorithm/submodule/models/classification/ote_efficientnet_b0_multilabel.yaml",
]

fp16 = dict(loss_scale=512.0)
