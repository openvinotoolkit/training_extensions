"""Supervised learning settings for video actor localization."""

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

optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=1e-5)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

lr_config = dict(
    policy="CosineAnnealing",
    by_epoch=False,
    min_lr=0,
    warmup="linear",
    warmup_by_epoch=True,
    warmup_iters=2,
    warmup_ratio=0.1,
)
checkpoint_config = dict(interval=1)
workflow = [("train", 1)]
evaluation = dict(interval=1, save_best="mAP@0.5IOU", final_metric="mAP@0.5IOU")
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
dist_params = dict(backend="nccl")
log_level = "INFO"
find_unused_parameters = False
# Temporary solution, gpu_ids is not used in otx
gpu_ids = [0]
seed = 2
