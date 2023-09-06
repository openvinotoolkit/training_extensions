"""Rotated FCOS Data pipeline."""
dataset_type = "OTXRotatedDataset"

data_root = "dota-coco/"

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (1024, 1024)

angle_version = "le90"

meta_keys = [
    "ori_filename",
    "flip_direction",
    "scale_factor",
    "img_norm_cfg",
    "gt_ann_ids",
    "flip",
    "ignored_labels",
    "ori_shape",
    "filename",
    "img_shape",
    "pad_shape",
]

train_pipeline = [
    dict(type="LoadImageFromOTXDataset", enable_memcache=True),
    dict(
        type="LoadAnnotationFromOTXDataset",
        domain="rotated_detection",
        with_bbox=True,
        with_angle=True,
        angle_version=angle_version,
    ),
    dict(type="RResize", img_scale=img_scale),
    dict(
        type="RRandomFlip",
        flip_ratio=[0.25, 0.25, 0.25],
        direction=["horizontal", "vertical", "diagonal"],
        version=angle_version,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"], meta_keys=meta_keys),
]

test_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
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
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        angle_version=angle_version,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        angle_version=angle_version,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        angle_version=angle_version,
        pipeline=test_pipeline,
    ),
)
