"""Data Pipeline of YOLOX model for Detection Task."""

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
__img_size = (640, 640)

train_pipeline = [
    dict(type="Mosaic", img_scale=__img_size, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.5, 1.5),
        border=(-__img_size[0] // 2, -__img_size[1] // 2),
    ),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Resize", img_scale=__img_size, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=114.0),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", size=(416, 416), pad_val=114.0),
            dict(type="Normalize", **__img_norm_cfg),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

__dataset_type = "CocoDataset"
__data_root = "data/coco/"
__samples_per_gpu = 2

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        # make sure to clean up recipe dataset
        _delete_=True,
        type="MultiImageMixDataset",
        dataset=dict(
            type=__dataset_type,
            ann_file=__data_root + "annotations/instances_train2017.json",
            img_prefix=__data_root + "train2017/",
            pipeline=[
                dict(type="LoadImageFromFile", to_float32=False),
                dict(type="LoadAnnotations", with_bbox=True),
            ],
        ),
        pipeline=train_pipeline,
    ),
    val=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_val2017.json",
        img_prefix=__data_root + "val2017/",
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_val2017.json",
        img_prefix=__data_root + "val2017/",
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
