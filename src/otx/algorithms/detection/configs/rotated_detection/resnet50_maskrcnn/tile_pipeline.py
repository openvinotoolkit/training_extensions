"""Tiling Pipeline of Resnet model for Instance-Seg Task."""

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

dataset_type = "CocoDataset"

img_size = (1344, 800)

tile_cfg = dict(
    tile_size=400, min_area_ratio=0.9, overlap_ratio=0.2, iou_threshold=0.45, max_per_img=1500, filter_empty_gt=True
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="Resize", img_scale=img_size, keep_ratio=False),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
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

__dataset_type = "CocoDataset"
__data_root = "data/coco/"

__samples_per_gpu = 4

train_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=__dataset_type,
        ann_file=__data_root + "annotations/instances_train.json",
        img_prefix=__data_root + "images/train",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
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
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
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
        pipeline=[dict(type="LoadImageFromFile")],
    ),
    pipeline=test_pipeline,
    **tile_cfg
)


data = dict(
    samples_per_gpu=__samples_per_gpu, workers_per_gpu=2, train=train_dataset, val=val_dataset, test=test_dataset
)
