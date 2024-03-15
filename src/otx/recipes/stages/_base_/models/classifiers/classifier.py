_base_ = "../model.py"

model = dict(
    type="ImageClassifier",
    task="classification",
    pretrained=None,
    backbone=dict(),
    head=dict(in_channels=-1, loss=dict(type="CrossEntropyLoss", loss_weight=1.0), topk=(1, 5)),
)

checkpoint_config = dict(type="CheckpointHookWithValResults")
