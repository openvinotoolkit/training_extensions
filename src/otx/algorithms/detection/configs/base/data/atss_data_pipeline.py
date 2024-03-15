"""Data Pipeline of ATSS model for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

__img_size = (992, 736)
__img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

train_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        load_ann_cfg=dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
        resize_cfg=dict(
            type="Resize",
            img_scale=(1088, 800),  # max sizes in random image scales
            keep_ratio=True,
            downscale_only=True,
        ),  # Resize to intermediate size if org image is bigger
        enable_memcache=True,  # Cache after resizing image & annotations
    ),
    dict(type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    dict(
        type="Resize",
        img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672), (992, 800)],
        multiscale_mode="value",
        keep_ratio=False,
        override=True,  # Allow multiple resize
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
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
        resize_cfg=dict(type="Resize", img_scale=__img_size, keep_ratio=False),
        enable_memcache=True,  # Cache after resizing image
    ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_size,
        flip=False,
        transforms=[
            dict(type="RandomFlip"),
            dict(type="Normalize", **__img_norm_cfg),
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
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="Normalize", **__img_norm_cfg),
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
