_base_ = [
  '../../../submodule/models/classification/ote_efficientnet_b0_multilabel.yaml',
]

fp16 = dict(loss_scale=512.)

custom_hooks = [
    dict(
        type='LazyEarlyStoppingHook',
        start=0,
        patience=10,
        iteration_patience=0,
        metric='mAP',
        interval=1,
        priority=75,
    ),
]
