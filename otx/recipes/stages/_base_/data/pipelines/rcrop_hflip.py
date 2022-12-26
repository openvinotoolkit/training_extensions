__img_norm_cfg = dict(mean=None, std=None)
__rcrop_target_size = -1
__rcrop_padding = -1

train_pipeline = [
    dict(type='RandomCrop', size=__rcrop_target_size, padding=__rcrop_padding),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **__img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='Normalize', **__img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
