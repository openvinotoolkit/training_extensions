"""Data Pipeline for Cls-Incr model of Segmentation Task."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name
__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
__img_scale = (544, 544)
__crop_size = (512, 512)

train_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        load_ann_cfg=dict(type="LoadAnnotationFromOTXDataset"),
        resize_cfg=dict(
            type="Resize",
            img_scale=__img_scale,
            downscale_only=True,
        ),  # Resize to intermediate size if org image is bigger
        enable_memcache=True,  # Cache after resizing image & annotations
    ),
    dict(type="Resize", img_scale=__img_scale, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=__crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="Pad", size=__crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_semantic_seg"],
        meta_keys=[
            "ori_shape",
            "pad_shape",
            "ori_filename",
            "filename",
            "scale_factor",
            "flip",
            "img_norm_cfg",
            "flip_direction",
            "ignored_labels",
            "img_shape",
        ],
    ),
]

val_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        resize_cfg=dict(type="Resize", img_scale=__img_scale, keep_ratio=False),
        enable_memcache=True,  # Cache after resizing image
    ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_scale,
        flip=False,
        transforms=[
            dict(type="RandomFlip"),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(
                type="Collect",
                keys=["img"],
            ),
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(
                type="Collect",
                keys=["img"],
            ),
        ],
    ),
]

data = dict(
    train=dict(type="OTXSegDataset", pipeline=train_pipeline),
    val=dict(type="OTXSegDataset", pipeline=val_pipeline),
    test=dict(type="OTXSegDataset", pipeline=test_pipeline),
)
