data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='OTXClsDataset',
        pipeline=[
            dict(type='Resize', size=224),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(type='AugMixAugment', config_str='augmix-m5-w3'),
            dict(type='RandomRotate', p=0.35, angle=(-10, 10)),
            dict(type='PILImageToNDArray', keys=['img']),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='OTXClsDataset',
        test_mode=True,
        pipeline=[
            dict(type='Resize', size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='OTXClsDataset',
        test_mode=True,
        pipeline=[
            dict(type='Resize', size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
