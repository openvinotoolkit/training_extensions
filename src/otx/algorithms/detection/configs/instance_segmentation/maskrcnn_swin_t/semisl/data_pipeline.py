"""Data Pipeline of Mask-RCNN ResNet50 model for Semi-Supervised Learning Instance Segmentation Task."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

# pylint: disable=invalid-name

# This is from otx/mpa/recipes/stages/_base_/data/pipelines/ubt.py
# This could be needed sync with incr-learning's data pipeline

__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

__img_size = (1344, 1344)

meta_keys = [
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
]

common_pipeline = [
    dict(type="Resize", img_scale=__img_size, keep_ratio=False),
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
    dict(type="LoadImageFromOTXDataset", enable_memcache=True),
    dict(
        type="LoadAnnotationFromOTXDataset",
        domain="instance_segmentation",
        with_bbox=True,
        with_mask=True,
        poly2mask=False,
    ),
    dict(type="Resize", img_scale=__img_size, keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"], meta_keys=meta_keys),
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
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(type="RepeatDataset", times=1, dataset=dict(type=__dataset_type, pipeline=train_pipeline)),
    val=dict(
        type=__dataset_type,
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=__dataset_type,
        test_mode=True,
        pipeline=test_pipeline,
    ),
    unlabeled=dict(
        type=__dataset_type,
        pipeline=unlabeled_pipeline,
    ),
)
