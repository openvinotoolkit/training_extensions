"""Data Pipeline of SupCon model for Classification Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
__resize_target_size = 224


__train_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        resize_cfg=dict(type="Resize", size=__resize_target_size, downscale_only=False),
        enable_memcache=True,  # Cache after resizing image
    ),
    dict(
        type="TwoCropTransform",
        pipeline=[
            dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
            dict(type="AugMixAugment", config_str="augmix-m5-w3"),
            dict(type="RandomRotate", p=0.35, angle=(-10, 10)),
            dict(type="ToNumpy"),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="ToTensor", keys=["gt_label"]),
            dict(type="Collect", keys=["img", "gt_label"]),
        ],
    ),
]

__val_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        resize_cfg=dict(type="Resize", size=__resize_target_size, downscale_only=False),
        enable_memcache=True,  # Cache after resizing image
    ),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

__test_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(type="Resize", size=__resize_target_size),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

__dataset_type = "OTXClsDataset"

data = dict(
    train=dict(type=__dataset_type, pipeline=__train_pipeline),
    val=dict(type=__dataset_type, test_mode=True, pipeline=__val_pipeline),
    test=dict(type=__dataset_type, test_mode=True, pipeline=__test_pipeline),
)
