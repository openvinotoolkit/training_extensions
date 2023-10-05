"""Data Pipeline of RTMDet-Inst model for Instance-Seg Task."""

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

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_size = (640, 640)

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

train_pipeline = [
    dict(type="LoadImageFromOTXDataset", enable_memcache=True),
    dict(
        type="LoadAnnotationFromOTXDataset",
        domain="instance_segmentation",
        with_bbox=True,
        with_mask=True,
        poly2mask=False,
    ),
    dict(type="CachedMosaic", img_scale=img_size, pad_val=114.0, max_cached_images=20, random_pop=False),
    dict(type="Resize", img_scale=(img_size[0] * 2, img_size[1] * 2), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=img_size),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Pad", size=img_size, pad_val=dict(img=(114, 114, 114))),
    dict(
        type="CachedMixUp",
        img_scale=img_size,
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"], meta_keys=meta_keys),
]

test_pipeline = [
    dict(type="LoadImageFromOTXDataset"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type="Resize", img_scale=img_size, keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", size=img_size, pad_val=114.0),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

dataset_type = "OTXDetDataset"

data = dict(
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
