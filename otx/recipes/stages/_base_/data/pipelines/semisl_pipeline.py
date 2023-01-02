img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
__resize_target_size = 224

common_pipeline = [
    dict(type="Resize", size=__resize_target_size),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="AugMixAugment", config_str="augmix-m5-w3"),
    dict(type="RandomRotate", p=0.35, angle=(-10, 10)),
]

train_pipeline = [
    *common_pipeline,
    dict(type="PILImageToNDArray", keys=["img"]),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]

unlabeled_pipeline = [
    *common_pipeline,
    dict(type="BranchImage", key_map=dict(img="img_weak")),
    dict(type="MPARandAugment", n=8, m=10),
    dict(type="BranchField", key_map=dict(img="img_weak")),
    dict(type="PILImageToNDArray", keys=["img", "img_weak"]),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img", "img_weak"]),
    dict(type="Collect", keys=["img", "img_weak"]),
]

test_pipeline = [
    dict(type="Resize", size=__resize_target_size),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
