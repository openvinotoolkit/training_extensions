"""Data Pipeline for Semi-Supervised Learning Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

# This is from otx/recipes/stages/_base_/data/pipelines/ubt.py
# This could be needed sync with incr-learning's data pipeline
__img_scale_test = (640, 640)
__img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)

common_pipeline = [
    dict(
        type="Resize",
        img_scale=[
            (692, 640),
            (596, 640),
            (896, 640),
            (692, 672),
            (692, 800),
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

# train_pipeline = [
#     dict(type="Mosaic", img_scale=__img_size, pad_val=114.0),
#     dict(
#         type="RandomAffine",
#         scaling_ratio_range=(0.1, 2),
#         border=(-__img_size[0] // 2, -__img_size[1] // 2),
#     ),
#     dict(type="MixUp", img_scale=__img_size, ratio_range=(0.8, 1.6), pad_val=114.0),
#     dict(type="YOLOXHSVRandomAug"),
#     dict(type="RandomFlip", flip_ratio=0.5),
#     dict(type="Resize", img_scale=__img_size, keep_ratio=True),
#     dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
#     dict(type="Normalize", **__img_norm_cfg),
#     dict(type="DefaultFormatBundle"),
#     dict(
#         type="Collect",
#         keys=["img", "gt_bboxes", "gt_labels"],
#         meta_keys=[
#             "ori_filename",
#             "flip_direction",
#             "scale_factor",
#             "img_norm_cfg",
#             "gt_ann_ids",
#             "flip",
#             "ignored_labels",
#             "ori_shape",
#             "filename",
#             "img_shape",
#             "pad_shape",
#         ],
#     ),
# ]

# train_pipeline = [
#     dict(type="LoadImageFromOTXDataset", enable_memcache=True),
#     dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
#     dict(type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
#     *common_pipeline,
#     dict(
#         type="Resize",
#         img_scale=__img_scale_test,
#         multiscale_mode="value",
#         keep_ratio=False,
#     ),
#     dict(type="RandomFlip", flip_ratio=0.5),
#     dict(type="Normalize", **__img_norm_cfg),
#     dict(type="DefaultFormatBundle"),
#     dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
# ]


train_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(type="LoadAnnotationFromOTXDataset", with_bbox=True),
    dict(type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    *common_pipeline,
    dict(type="ToTensor", keys=["gt_bboxes", "gt_labels"]),
    dict(
        type="ToDataContainer",
        fields=[
            dict(key="img", stack=True),
            dict(key="img0", stack=True),
            dict(key="gt_bboxes"),
            dict(key="gt_labels"),
        ],
    ),
    dict(
        type="Collect",
        keys=["img", "img0", "gt_bboxes", "gt_labels"],
    ),
]


unlabeled_pipeline = [
    dict(type="LoadImageFromOTXDataset", enable_memcache=True),
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

test_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_scale_test,
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
        pipeline=test_pipeline,
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
