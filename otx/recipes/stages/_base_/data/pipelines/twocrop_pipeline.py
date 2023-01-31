img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
__resize_target_size = 224


train_pipeline = [
    dict(
        type="TwoCropTransform",
        pipeline=[
            dict(type="Resize", size=__resize_target_size),
            dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
            dict(type="AugMixAugment", config_str="augmix-m5-w3"),
            dict(type="RandomRotate", p=0.35, angle=(-10, 10)),
            dict(type="ToNumpy"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="ToTensor", keys=["gt_label"]),
            dict(type="Collect", keys=["img", "gt_label"]),
        ],
    )
]

test_pipeline = [
    dict(type="Resize", size=__resize_target_size),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
