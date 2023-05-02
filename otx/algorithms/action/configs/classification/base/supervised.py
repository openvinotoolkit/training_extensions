"""Base supervised learning recipe for video classification."""

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

seed = 2

evaluation = dict(interval=1, metrics=["top_k_accuracy", "mean_class_accuracy"], final_metric="mean_class_accuracy")

optimizer = dict(
    type="AdamW",
    lr=0.001,
    weight_decay=0.0001,
)

optimizer_config = dict(grad_clip=dict(max_norm=40.0, norm_type=2))
lr_config = dict(
    policy="step",
    step=5,
    warmup="linear",
    warmup_by_epoch=True,
    warmup_iters=3,
)

# runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook", ignore_last=False),
    ],
)
# runtime settings
log_level = "INFO"
workflow = [("train", 1)]

find_unused_parameters = False
gpu_ids = range(0, 1)

dist_params = dict(backend="nccl")
