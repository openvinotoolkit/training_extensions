model = dict(
    type='SAMImageClassifier',
    backbone=dict(type='otx.OTXEfficientNet', pretrained=True, version='b0'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='CustomMultiLabelLinearClsHead',
        num_classes=1000,
        in_channels=-1,
        loss=dict(
            type='AsymmetricAngularLossWithIgnore',
            loss_weight=1.0,
            gamma_pos=0.0,
            gamma_neg=1.0,
            reduction='sum'),
        topk=(1, 5),
        normalized=True,
        scale=7.0),
    task='classification',
    pretrained=None)
dist_params = dict(backend='nccl', linear_scale_lr=True)
cudnn_benchmark = True
seed = 5
deterministic = False
hparams = dict(dummy=0)
task_adapt = dict(op='REPLACE', type='mpa')
log_level = 'INFO'
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', ignore_last=False),
        dict(type='TensorboardLoggerHook')
    ])
optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None, type='SAMOptimizerHook')
runner = dict(type='EpochRunnerWithCancel', max_epochs=50)
workflow = [('train', 1)]
lr_config = dict(
    policy='OneCycle',
    pct_start=0.200001,
    div_factor=100,
    final_div_factor=1000)
evaluation = dict(metric=['accuracy', 'class_accuracy'])
load_from = None
resume_from = None
checkpoint_config = dict(
    interval=1, max_keep_ckpts=1, type='CheckpointHookWithValResults')
custom_hooks = [dict(type='ModelEmaV2Hook')]
fp16 = dict(loss_scale=512.0)
