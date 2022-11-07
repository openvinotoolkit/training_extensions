_base_ = [
    "../../../submodule/samples/cfgs/models/backbones/ote_mobilenet_v2_w1.yaml",
    "../../../submodule/recipes/stages/_base_/models/detectors/ssd.custom.py",
]

model = dict(
    backbone=dict(
        out_indices=(
            4,
            5,
        )
    )
)
fp16 = dict(loss_scale=512.0)
