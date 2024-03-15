"""Tiling Pipeline of YOLOX variants for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

# NOTE: SKIP MOSAIC AND MultiImageMixDataset in tiling

img_scale = (640, 640)

tile_cfg = dict(
    tile_size=400, min_area_ratio=0.9, overlap_ratio=0.2, iou_threshold=0.45, max_per_img=1500, filter_empty_gt=True
)

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=[
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
        ],
    ),
]

test_pipeline = [
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

__dataset_type = "OTXDetDataset"

train_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=__dataset_type,
        pipeline=[
            dict(type="LoadImageFromOTXDataset", enable_memcache=True),
            dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
        ],
    ),
    pipeline=train_pipeline,
    **tile_cfg
)

val_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=__dataset_type,
        pipeline=[
            dict(type="LoadImageFromOTXDataset", enable_memcache=True),
            dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
        ],
    ),
    pipeline=test_pipeline,
    **tile_cfg
)

test_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=__dataset_type,
        test_mode=True,
        pipeline=[dict(type="LoadImageFromOTXDataset")],
    ),
    pipeline=test_pipeline,
    **tile_cfg
)


data = dict(train=train_dataset, val=val_dataset, test=test_dataset)
