"""Data Pipeline of Class-Incr model for Classification Task."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
__resize_target_size = 224

__train_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        resize_cfg=dict(type="Resize", size=__resize_target_size, downscale_only=True),
        # To be resized in this op only if input is larger than expected size
        # for speed & cache memory efficiency.
        enable_memcache=True,  # Cache after resizing image
    ),
    dict(type="RandomResizedCrop", size=__resize_target_size, efficientnet_style=True),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(
        type="Collect",
        keys=["img", "gt_label"],
        meta_keys=[
            "flip_direction",
            "entity_id",
            "ori_filename",
            "filename",
            "img_norm_cfg",
            "img_shape",
            "label_id",
            "pad_shape",
            "scale_factor",
            "flip",
            "ori_shape",
            "ignored_labels",
        ],
    ),
]

__val_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        resize_cfg=dict(type="Resize", size=__resize_target_size),
        enable_memcache=True,  # Cache after resizing image
    ),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

__test_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(type="ResizeTo", size=__resize_target_size),
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
