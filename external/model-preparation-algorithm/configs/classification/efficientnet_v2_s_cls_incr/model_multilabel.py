_base_ = [
  '../../../submodule/models/classification/ote_efficientnet_v2_s_multilabel.yaml',
]

optimizer = dict(lr=0.025)
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
