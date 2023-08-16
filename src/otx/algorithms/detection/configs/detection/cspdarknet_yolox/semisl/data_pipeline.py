"""Data Pipeline of YOLOX model for Semi-Supervised Learning Detection Task."""

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

# This is from src/otx/mpa/recipes/stages/_base_/data/pipelines/ubt.py
# This could be needed sync with incr-learning's data pipeline
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
    dict(type="LoadImageFromOTXDataset", enable_memcache=True),
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
