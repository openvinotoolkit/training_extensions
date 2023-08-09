"""Data Pipeline of Class-Incr model for Classification Task."""

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
__resize_target_size = 224

__train_pipeline = [
    dict(type="RandomResizedCrop", size=__resize_target_size, efficientnet_style=True),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]

__test_pipeline = [
    dict(type="Resize", size=__resize_target_size),
    dict(type="Normalize", **__img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

__dataset_type = "OTXClsDataset"
__samples_per_gpu = 32
__workers_per_gpu = 2

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=__workers_per_gpu,
    train=dict(type=__dataset_type, pipeline=__train_pipeline),
    val=dict(type=__dataset_type, test_mode=True, pipeline=__test_pipeline),
    test=dict(type=__dataset_type, test_mode=True, pipeline=__test_pipeline),
)
