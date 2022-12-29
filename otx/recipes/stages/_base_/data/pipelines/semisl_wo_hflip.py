img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
__resize_target_size = 224

train_pipeline = [
    dict(type="Resize", size=__resize_target_size),
    dict(type="AugMixAugment", config_str="augmix-m5-w3"),
    dict(type="RandomRotate", p=0.35, angle=(-10, 10)),
    dict(type="ToNumpy"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
test_pipeline = [
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
train_pipeline_strong = [
    dict(type="Resize", size=__resize_target_size),
    dict(type="MPARandAugment", n=2, m=10),
    dict(type="AugMixAugment", config_str="augmix-m5-w3"),
    dict(type="RandomRotate", p=0.35, angle=(-10, 10)),
    dict(type="ToNumpy"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
