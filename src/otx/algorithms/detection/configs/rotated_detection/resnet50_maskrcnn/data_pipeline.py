"""Data Pipeline of Resnet model for Rotated-Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

__img_size = (1024, 1024)

# TODO: A comparison experiment is needed to determine which value is appropriate for to_rgb.
__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        load_ann_cfg=dict(
            type="LoadAnnotationFromOTXDataset",
            domain="rotated_detection",
            with_bbox=True,
            with_mask=True,
            poly2mask=False,
        ),
        resize_cfg=dict(
            type="Resize",
            img_scale=__img_size,
            keep_ratio=True,
        ),
        enable_memcache=True,  # Cache after resizing image & annotations
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
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
        resize_cfg=dict(type="Resize", img_scale=__img_size, keep_ratio=True),
        enable_memcache=True,  # Cache after resizing image
    ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_size,
        flip=False,
        transforms=[
            dict(type="RandomFlip"),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
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
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

__dataset_type = "OTXDetDataset"

data = dict(
    train=dict(
        type=__dataset_type,
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
