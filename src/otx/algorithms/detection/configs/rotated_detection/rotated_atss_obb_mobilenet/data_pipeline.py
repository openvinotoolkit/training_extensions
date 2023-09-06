"""Data Pipeline of rotated_atss_obb_mobilenet for Rotated-Detection Task."""
dataset_type = "OTXRotatedDataset"

data_root = "dota-coco/"

img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

img_scale = (992, 736)

angle_version = "le135"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_angle=True, angle_version=angle_version),
    dict(
        type="RResize",
        img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672), (992, 800)],
        multiscale_mode="value",
    ),
    dict(
        type="RRandomFlip",
        flip_ratio=[0.25, 0.25, 0.25],
        direction=["horizontal", "vertical", "diagonal"],
        version=angle_version,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="RResize"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        angle_version=angle_version,
        ann_file=data_root + "anno/DOTA_train.json",
        img_prefix=data_root + "train/images-jpeg/",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        angle_version=angle_version,
        ann_file=data_root + "anno/DOTA_val.json",
        img_prefix=data_root + "val/images-jpeg/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        angle_version=angle_version,
        ann_file=data_root + "anno/DOTA_val.json",
        img_prefix=data_root + "val/images-jpeg/",
        pipeline=test_pipeline,
    ),
)
