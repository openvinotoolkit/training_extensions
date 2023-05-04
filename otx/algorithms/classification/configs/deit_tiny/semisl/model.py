"""deit-tiny config for semi-supervised multi-class classification."""

# pylint: disable=invalid-name

_base_ = ["../../../../../recipes/stages/classification/semisl.yaml", "../../base/models/deit.py"]
ckpt_url = "https://download.openmmlab.com/mmclassification/v0/deit/deit-tiny_pt-4xb256_in1k_20220218-13b382a0.pth"

model = dict(
    type="SemiSLClassifier",
    task="classification",
    backbone=dict(arch="deit-tiny", init_cfg=dict(type="Pretrained", checkpoint=ckpt_url, prefix="backbone")),
    head=dict(
        type="SemiLinearClsHead",
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
        ),
    ),
)

fp16 = dict(loss_scale=512.0)
