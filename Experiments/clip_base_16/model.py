clip_ckpt = "../vit-b-16-clip.pth"
model = dict(
    type='SAMImageClassifier',
    backbone=dict(
        arch='base',
        drop_path_rate=0.0,
        drop_rate=0.1,
        final_norm=True,
        img_size=224,
        in_channels=3,
        init_cfg=dict(type='Pretrained', checkpoint=clip_ckpt, prefix='backbone'),
        interpolate_mode='bicubic',
        layer_cfgs=dict(),
        norm_cfg=dict(eps=1e-06, type='LN'),
        out_indices=-1,
        output_cls_token=True,
        patch_cfg=dict(bias=False),
        patch_size=16,
        qkv_bias=True,
        type='CLIPVisionTransformer',
        with_cls_token=True),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=-1,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)),
    task='classification',
    pretrained=None)
dist_params = dict(backend='nccl', linear_scale_lr=True)
cudnn_benchmark = True
seed = 0
deterministic = True
hparams = dict(dummy=0)
task_adapt = dict(op='REPLACE', type='mpa')
log_level = 'INFO'
log_config = dict(
    interval=1,
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
fp16 = dict(loss_scale=512.0)
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
