__img_scale = (992, 736)
__img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

common_pipeline = [
    dict(
        type="Resize",
        img_scale=[
            (992, 736),
            (896, 736),
            (1088, 736),
            (992, 672),
            (992, 800),
        ],
        multiscale_mode="value",
        keep_ratio=False,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="BranchImage", key_map=dict(img="img0")),
    dict(type="NDArrayToPILImage", keys=["img"]),
    dict(
        type="RandomApply",
        transform_cfgs=[
            dict(
                type="ColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            )
        ],
        p=0.8,
    ),
    dict(type="RandomGrayscale", p=0.2),
    dict(
        type="RandomApply",
        transform_cfgs=[
            dict(
                type="RandomGaussianBlur",
                sigma_min=0.1,
                sigma_max=2.0,
            )
        ],
        p=0.5,
    ),
    dict(type="PILImageToNDArray", keys=["img"]),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="NDArrayToTensor", keys=["img", "img0"]),
    dict(
        type="RandomErasing",
        p=0.7,
        scale=[0.05, 0.2],
        ratio=[0.3, 3.3],
        value="random",
    ),
    dict(
        type="RandomErasing",
        p=0.5,
        scale=[0.02, 0.2],
        ratio=[0.10, 6.0],
        value="random",
    ),
    dict(
        type="RandomErasing",
        p=0.3,
        scale=[0.02, 0.2],
        ratio=[0.05, 8.0],
        value="random",
    ),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    *common_pipeline,
    dict(type="ToTensor", keys=["gt_bboxes", "gt_labels"]),
    dict(
        type="ToDataContainer",
        fields=[
            dict(key="img", stack=True),
            dict(key="img0", stack=True),
            dict(key="gt_bboxes"),
            dict(key="gt_labels"),
        ],
    ),
    dict(
        type="Collect",
        keys=["img", "img0", "gt_bboxes", "gt_labels"],
    ),
]

unlabeled_pipeline = [
    dict(type="LoadImageFromFile"),
    *common_pipeline,
    dict(
        type="ToDataContainer",
        fields=[
            dict(key="img", stack=True),
            dict(key="img0", stack=True),
        ],
    ),
    dict(
        type="Collect",
        keys=[
            "img",
            "img0",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
