"""Data pipeline for DETR based models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

# dataset settings
dataset_type = "OTXDetDataset"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        load_ann_cfg=dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
        resize_cfg=dict(
            type="Resize",
            img_scale=(1333, 800),  # max sizes in random image scales
            keep_ratio=True,
            downscale_only=True,
        ),  # Resize to intermediate size if org image is bigger
        enable_memcache=True,  # Cache after resizing image & annotations
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                    override=True,  # Allows multiple resize
                )
            ],
            [
                dict(
                    type="Resize",
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode="value",
                    keep_ratio=True,
                    override=True,  # Allows multiple resize
                ),
                dict(type="RandomCrop", crop_type="absolute_range", crop_size=(384, 600), allow_negative_crop=True),
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                    override=True,  # Allows multiple resize
                ),
            ],
        ],
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=1),
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
            img_scale=(1333, 800),
            keep_ratio=True,
            downscale_only=False,
        ),
        enable_memcache=True,  # Cache after resizing image & annotations
    ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=1),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=1),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    train=dict(
        type=dataset_type,
        filter_empty_gt=False,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
    ),
)
