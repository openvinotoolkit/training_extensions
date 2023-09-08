"""Data Pipeline for Cls-Incr model of Segmentation Task."""

# Copyright (C) 2022 Intel Corporation
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
__img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
__img_scale = (544, 544)
__crop_size = (512, 512)

train_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        load_ann_cfg=dict(type="LoadAnnotationFromOTXDataset", use_otx_adapter=True),
        resize_cfg=dict(
            type="Resize",
            img_scale=__img_scale,
            downscale_only=True,
        ),  # Resize to intermediate size if org image is bigger
        enable_memcache=True,  # Cache after resizing image & annotations
    ),
    dict(type="Resize", img_scale=__img_scale, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=__crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="Pad", size=__crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_semantic_seg"],
        meta_keys=[
            "ori_shape",
            "pad_shape",
            "ori_filename",
            "filename",
            "scale_factor",
            "flip",
            "img_norm_cfg",
            "flip_direction",
            "ignored_labels",
            "img_shape",
        ],
    ),
]

val_pipeline = [
    dict(
        type="LoadResizeDataFromOTXDataset",
        resize_cfg=dict(type="Resize", img_scale=__img_scale, keep_ratio=False),
        enable_memcache=True,  # Cache after resizing image
        use_otx_adapter=True,
    ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_scale,
        flip=False,
        transforms=[
            dict(type="RandomFlip"),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(
                type="Collect",
                keys=["img"],
            ),
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromOTXDataset", use_otx_adapter=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=__img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(
                type="Collect",
                keys=["img"],
            ),
        ],
    ),
]

data = dict(
    train=dict(type="OTXSegDataset", pipeline=train_pipeline),
    val=dict(type="OTXSegDataset", pipeline=val_pipeline),
    test=dict(type="OTXSegDataset", pipeline=test_pipeline),
)
