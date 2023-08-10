"""Data Configuration of VFNet model for Detection Task."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

__train_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
    dict(type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.1),
    dict(
        type="Resize",
        img_scale=[(1344, 480), (1344, 960)],
        multiscale_mode="range",
        keep_ratio=False,
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
__test_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1344, 800),
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

__dataset_type = "OTXDetDataset"
__data_root = "data/coco/"

__samples_per_gpu = 2

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_train2017.json",
        img_prefix=__data_root + "train2017/",
        pipeline=__train_pipeline,
    ),
    val=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_val2017.json",
        img_prefix=__data_root + "val2017/",
        test_mode=True,
        pipeline=__test_pipeline,
    ),
    test=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_val2017.json",
        img_prefix=__data_root + "val2017/",
        test_mode=True,
        pipeline=__test_pipeline,
    ),
)
