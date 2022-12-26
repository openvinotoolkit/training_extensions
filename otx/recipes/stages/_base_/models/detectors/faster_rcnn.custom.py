_base_ = [
    './faster_rcnn.py'
]

model= dict(
    type='CustomTwoStageDetector',
)

__img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

data = dict(
    pipeline_options=dict(
        MinIouRandomCrop=dict(min_crop_size=0.1),
        Resize=dict(img_scale=(1333, 800), keep_ratio=True),
        Normalize=dict(**__img_norm_cfg),
        MultiScaleFlipAug=dict(
            img_scale=(1333, 800),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False),
                dict(type='Normalize', **__img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ],
        ),
    ),
)

