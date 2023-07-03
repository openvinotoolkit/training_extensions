"""deit-tiny for multi-label config."""

# pylint: disable=invalid-name

_base_ = ["../../../../recipes/stages/classification/multilabel/incremental.yaml", "../base/models/deit.py"]
ckpt_url = "https://download.openmmlab.com/mmclassification/v0/deit/deit-tiny_pt-4xb256_in1k_20220218-13b382a0.pth"

model = dict(
    type="SAMImageClassifier",
    task="classification",
    backbone=dict(arch="deit-tiny", init_cfg=dict(type="Pretrained", checkpoint=ckpt_url, prefix="backbone")),
    head=dict(
        type="CustomMultiLabelLinearClsHead",
        loss=dict(type="AsymmetricLossWithIgnore"),
    ),
)

fp16 = dict(loss_scale=512.0)

optimizer = dict(_delete_=True, type="AdamW", lr=0.01, weight_decay=0.05)
optimizer_config = dict(_delete_=True)
