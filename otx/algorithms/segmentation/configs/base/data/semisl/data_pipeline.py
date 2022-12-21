"""Data Pipeline for Semi-SL model of Segmentation Task."""

__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
__img_scale = (544, 544)
__crop_size = (512, 512)

__common_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=__img_scale, ratio_range=(0.5, 2.0), keep_ratio=False),
    dict(type="RandomCrop", crop_size=__crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomRotate", prob=0.5, degree=30, pad_val=0, seg_pad_val=255),
]

train_pipeline = [
    *__common_pipeline,
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="Pad", size=__crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

unlabeled_pipeline = [
    *__common_pipeline,
    dict(type="Pad", size=__crop_size, pad_val=0, seg_pad_val=255),
    dict(type="BranchImage", key_map=dict(img="ul_w_img")),
    dict(
        type="ProbCompose",
        probs=[0.7, 0.3],
        transforms=[
            dict(
                type="PhotoMetricDistortion",
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18,
            ),
            dict(type="RGB2Gray"),
        ],
    ),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "ul_w_img"]),
]

# TODO (Soobee) : Remove Repeatdataset in data config
data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    unlabeled=dict(pipeline=unlabeled_pipeline),
)
