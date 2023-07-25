"""Data Pipeline of Mask2Former model for Instance-Seg Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name


# TODO: A comparison experiment is needed to determine which value is appropriate for to_rgb.
__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)

__img_size = (1024, 1024)
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Resize", img_scale=__img_size, ratio_range=(0.1, 2.0), multiscale_mode="range", keep_ratio=True),
    dict(type="RandomCrop", crop_size=__img_size, crop_type="absolute", recompute_bbox=True, allow_negative_crop=True),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-05, 1e-05), by_mask=True),
    dict(type="Pad", size=__img_size, pad_val=pad_cfg),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="DefaultFormatBundle", img_to_float=True),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_size,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", size_divisor=32, pad_val=pad_cfg),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

__dataset_type = "CocoDataset"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=__dataset_type,
        ann_file="data/coco/annotations/instances_train2017.json",
        img_prefix="data/coco/train2017",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=__dataset_type,
        test_mode=True,
        ann_file="data/coco/annotations/instances_val2017.json",
        img_prefix="data/coco/val2017",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=__dataset_type,
        test_mode=True,
        ann_file="data/coco/annotations/instances_val2017.json",
        img_prefix="data/coco/val2017",
        pipeline=test_pipeline,
    ),
)
