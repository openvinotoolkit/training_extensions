"""Data Pipeline of YOLOX variants for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

__img_size = (640, 640)
__img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type="Mosaic", img_scale=__img_size, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        border=(-__img_size[0] // 2, -__img_size[1] // 2),
    ),
    dict(type="MixUp", img_scale=__img_size, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Resize", img_scale=__img_size, keep_ratio=True, override=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
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
            img_scale=__img_size,
            keep_ratio=True,
            downscale_only=False,
        ),
        enable_memcache=True,  # Cache after resizing image
    ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_size,
        flip=False,
        transforms=[
            dict(type="RandomFlip"),
            dict(type="Pad", size=__img_size, pad_val=114.0),
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
        img_scale=__img_size,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", size=__img_size, pad_val=114.0),
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
