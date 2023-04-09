model = dict(
    type='SAMImageClassifier',
    backbone=dict(
        type='otx.OTXMobileNetV3',
        pretrained=True,
        mode='large',
        width_mult=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='CustomHierarchicalNonLinearClsHead',
        num_classes=1000,
        in_channels=960,
        hid_channels=1280,
        act_cfg=dict(type='HSwish'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        multilabel_loss=dict(
            type='AsymmetricLossWithIgnore', gamma_pos=0.0, gamma_neg=4.0)),
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
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None, type='SAMOptimizerHook')
runner = dict(type='EpochRunnerWithCancel', max_epochs=20)
workflow = [('train', 1)]
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=0.0001)
evaluation = dict(metric=['accuracy', 'class_accuracy'])
load_from = None
resume_from = None
checkpoint_config = dict(
    interval=1, max_keep_ckpts=1, type='CheckpointHookWithValResults')
