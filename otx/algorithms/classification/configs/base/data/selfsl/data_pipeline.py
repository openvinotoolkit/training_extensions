"""Data Pipeline of Self-SL model for Classification Task."""

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
__img_size = 224

__train_pipeline_v0 = [
    dict(type="RandomResizedCrop", size=__img_size),
    dict(type="RandomFlip"),
    dict(
        type="RandomAppliedTrans",
        transforms=[
            dict(type="OTXColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        ],
        p=0.8,
    ),
    dict(type="RandomGrayscale", gray_prob=0.2),
    dict(type="GaussianBlur", sigma_min=0.1, sigma_max=2.0),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
__train_pipeline_v1 = [
    dict(type="RandomResizedCrop", size=__img_size),
    dict(type="RandomFlip"),
    dict(
        type="RandomAppliedTrans",
        transforms=[
            dict(type="OTXColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        ],
        p=0.8,
    ),
    dict(type="RandomGrayscale", gray_prob=0.2),
    dict(type="RandomAppliedTrans", transforms=[dict(type="GaussianBlur", sigma_min=0.1, sigma_max=2.0)], p=0.1),
    dict(type="RandomAppliedTrans", transforms=[dict(type="Solarize", thr=128)], p=0.2),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

data = dict(
    train=dict(pipeline=dict(view0=__train_pipeline_v0, view1=__train_pipeline_v1)),
)
