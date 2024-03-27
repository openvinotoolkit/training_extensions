"""Data Pipeline of YOLOX Tiny model for Detection Task."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
__img_size = (640, 640)
__img_size_test = (416, 416)

train_pipeline = [
    dict(type="Mosaic", img_scale=__img_size, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.5, 1.5),
        border=(-__img_size[0] // 2, -__img_size[1] // 2),
    ),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Resize", img_scale=__img_size, keep_ratio=True, override=True),  # Allow multiple resize
    dict(type="Pad", pad_to_square=True, pad_val=114.0),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=[
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
        ],
    ),
]

val_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        resize_cfg=dict(
            type="Resize",
            img_scale=__img_size_test,
            keep_ratio=True,
            downscale_only=False,
        ),
        enable_memcache=True,  # Cache after resizing image
    ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_size_test,
        flip=False,
        transforms=[
            dict(type="RandomFlip"),
            dict(type="Pad", size=__img_size_test, pad_val=114.0),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_size_test,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", size=__img_size_test, pad_val=114.0),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

__dataset_type = "OTXDetDataset"

data = dict(
    train=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            type=__dataset_type,
            pipeline=[
                dict(
                    type="LoadResizeDataFromOTXDataset",
                    load_ann_cfg=dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
                    resize_cfg=dict(
                        type="Resize",
                        img_scale=__img_size,
                        keep_ratio=True,
                        downscale_only=True,
                    ),  # Resize to intermediate size if org image is bigger
                    to_float32=False,
                    enable_memcache=True,  # Cache after resizing image & annotations
                ),
            ],
        ),
        pipeline=train_pipeline,
    ),
    val=dict(
        type=__dataset_type,
        test_mode=True,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=__dataset_type,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
