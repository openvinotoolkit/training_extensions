"""Data Pipeline with angle version le135 for Rotated-Detection Task."""
dataset_type = "OTXRotatedDataset"

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (1024, 1024)

tile_cfg = dict(
    tile_size=400, min_area_ratio=0.9, overlap_ratio=0.2, iou_threshold=0.45, max_per_img=1500, filter_empty_gt=True
)

angle_version = "le90"

meta_keys = [
    "filename",
    "ori_filename",
    "ori_shape",
    "img_shape",
    "pad_shape",
    "scale_factor",
    "flip",
    "flip_direction",
    "img_norm_cfg",
]

train_pipeline = [
    dict(type="RResize", img_scale=img_scale),
    dict(type="RRandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=meta_keys,
    ),
]


test_pipeline = [
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

train_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=dataset_type,
        angle_version=angle_version,
        pipeline=[
            dict(type="LoadImageFromOTXDataset", enable_memcache=True),
            dict(
                type="LoadAnnotationFromOTXDataset",
                domain="rotated_detection",
                with_bbox=True,
                with_angle=True,
                angle_version=angle_version,
            ),
        ],
    ),
    pipeline=train_pipeline,
    **tile_cfg
)

val_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=dataset_type,
        angle_version=angle_version,
        pipeline=[
            dict(type="LoadImageFromOTXDataset", enable_memcache=True),
            dict(
                type="LoadAnnotationFromOTXDataset",
                domain="rotated_detection",
                with_bbox=True,
                with_angle=True,
                angle_version=angle_version,
            ),
        ],
    ),
    pipeline=test_pipeline,
    **tile_cfg
)

test_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=dataset_type,
        angle_version=angle_version,
        test_mode=True,
        pipeline=[dict(type="LoadImageFromOTXDataset")],
    ),
    pipeline=test_pipeline,
    **tile_cfg
)


data = dict(train=train_dataset, val=val_dataset, test=test_dataset)
