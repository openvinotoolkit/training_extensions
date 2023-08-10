"""Tiling Pipeline of EfficientNetB2B model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

dataset_type = "OTXDetDataset"

img_size = (512, 512)

tile_cfg = dict(
    tile_size=400, min_area_ratio=0.9, overlap_ratio=0.2, iou_threshold=0.45, max_per_img=1500, filter_empty_gt=True
)

img_norm_cfg = dict(mean=(103.53, 116.28, 123.675), std=(1.0, 1.0, 1.0), to_rgb=True)

train_pipeline = [
    dict(type="Resize", img_scale=img_size, keep_ratio=False),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
        meta_keys=[
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
        ],
    ),
]

test_pipeline = [
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    )
]

__dataset_type = "OTXDetDataset"
__data_root = "data/coco/"

__samples_per_gpu = 4

train_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_train.json",
        img_prefix=__data_root + "images/train",
        pipeline=[
            dict(type="LoadImageFromOTXDataset", enable_memcache=True),
            dict(type="LoadAnnotationFromOTXDataset", domain="instance_segmentation", with_bbox=True, with_mask=True),
        ],
    ),
    pipeline=train_pipeline,
    **tile_cfg
)

val_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_val.json",
        img_prefix=__data_root + "images/val",
        pipeline=[
            dict(type="LoadImageFromOTXDataset", enable_memcache=True),
            dict(type="LoadAnnotationFromOTXDataset", domain="instance_segmentation", with_bbox=True, with_mask=True),
        ],
    ),
    pipeline=test_pipeline,
    **tile_cfg
)

test_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_test.json",
        img_prefix=__data_root + "images/test",
        test_mode=True,
        pipeline=[dict(type="LoadImageFromOTXDataset")],
    ),
    pipeline=test_pipeline,
    **tile_cfg
)


data = dict(
    samples_per_gpu=__samples_per_gpu, workers_per_gpu=2, train=train_dataset, val=val_dataset, test=test_dataset
)
