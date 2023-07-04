"""MobileNet-V3-Small for self-supervised learning config."""

# pylint: disable=invalid-name

_base_ = ["../../../../../recipes/stages/classification/selfsl.yaml", "../../base/models/mobilenet_v3.py"]

model = dict(
    type="BYOL",
    task="classification",
    base_momentum=0.996,
    neck=dict(type="SelfSLMLP", in_channels=576, hid_channels=4096, out_channels=256, with_avg_pool=True),
    head=dict(
        _delete_=True,
        type="ConstrastiveHead",
        predictor=dict(type="SelfSLMLP", in_channels=256, hid_channels=4096, out_channels=256, with_avg_pool=False),
    ),
)

load_from = None

resume_from = None

fp16 = None
