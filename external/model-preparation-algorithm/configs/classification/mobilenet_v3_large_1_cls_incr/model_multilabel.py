_base_ = [
  '../../../submodule/models/classification/ote_mobilenet_v3_large_multilabel.yaml',
]

runner = dict(max_epochs=90)
fp16 = dict(loss_scale=512.)

custom_hooks = [
    dict(
        type='LazyEarlyStoppingHook',
        start=0,
        patience=10,
        iteration_patience=0,
        metric='mAP',
        interval=1,
        priority='ABOVE_NORMAL',
    ),
]
