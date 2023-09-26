"""Data Pipeline of YOLOX Tiny model for Semi-Supervised Learning Detection Task."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

# This is from src/otx/recipes/stages/_base_/data/pipelines/ubt.py
# This could be needed sync with incr-learning's data pipeline
_base_ = ["../../../base/data/semisl/base_semisl_det_data_pipeline.py"]

__img_scale = (992, 736)
__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

common_pipeline = [
    dict(
        type="Resize",
        img_scale=[
            (992, 736),
            (896, 736),
            (1088, 736),
            (992, 672),
            (992, 800),
        ],
        multiscale_mode="value",
        keep_ratio=False,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="BranchImage", key_map=dict(img="img0")),
    dict(type="NDArrayToPILImage", keys=["img"]),
    dict(
        type="RandomApply",
        transform_cfgs=[
            dict(
                type="ColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            )
        ],
        p=0.8,
    ),
    dict(type="RandomGrayscale", p=0.2),
    dict(
        type="RandomApply",
        transform_cfgs=[
            dict(
                type="RandomGaussianBlur",
                sigma_min=0.1,
                sigma_max=2.0,
            )
        ],
        p=0.5,
    ),
    dict(type="PILImageToNDArray", keys=["img"]),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="NDArrayToTensor", keys=["img", "img0"]),
    dict(
        type="RandomErasing",
        p=0.7,
        scale=[0.05, 0.2],
        ratio=[0.3, 3.3],
        value="random",
    ),
    dict(
        type="RandomErasing",
        p=0.5,
        scale=[0.02, 0.2],
        ratio=[0.10, 6.0],
        value="random",
    ),
    dict(
        type="RandomErasing",
        p=0.3,
        scale=[0.02, 0.2],
        ratio=[0.05, 8.0],
        value="random",
    ),
]

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
        img_scale=[
            (992, 736),
            (896, 736),
            (1088, 736),
            (992, 672),
            (992, 800),
        ],
        multiscale_mode="value",
        keep_ratio=False,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

unlabeled_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        resize_cfg=dict(
            type="Resize",
            img_scale=(1088, 800),  # max sizes in random image scales
            keep_ratio=True,
            downscale_only=True,
        ),  # Resize to intermediate size if org image is bigger
        enable_memcache=True,  # Cache after resizing image & annotations
    ),
    *common_pipeline,
    dict(
        type="ToDataContainer",
        fields=[
            dict(key="img", stack=True),
            dict(key="img0", stack=True),
        ],
    ),
    dict(
        type="Collect",
        keys=[
            "img",
            "img0",
        ],
    ),
]

val_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        resize_cfg=dict(
            type="Resize",
            img_scale=__img_scale,
            keep_ratio=False,
            downscale_only=False,
        ),
        enable_memcache=True,  # Cache after resizing image & annotations
    ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_scale,
        flip=False,
        transforms=[
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
        img_scale=__img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    train=dict(
        type="OTXDetDataset",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="OTXDetDataset",
        pipeline=val_pipeline,
    ),
    test=dict(
        type="OTXDetDataset",
        pipeline=test_pipeline,
    ),
    unlabeled=dict(
        type="OTXDetDataset",
        pipeline=unlabeled_pipeline,
    ),
)
