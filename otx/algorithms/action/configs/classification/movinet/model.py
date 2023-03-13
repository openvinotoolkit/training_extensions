"""Model configuration of MoViNet model for Action Classification Task."""

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

num_classes = 400
model = dict(
    type="MoViNetRecognizer",
    backbone=dict(type="OTXMoViNet"),
    cls_head=dict(
        type="MoViNetHead",
        in_channels=480,
        hidden_dim=2048,
        num_classes=num_classes,
        loss_cls=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips="prob"),
)


evaluation = dict(interval=1, metrics=["top_k_accuracy", "mean_class_accuracy"], final_metric="mean_class_accuracy")

optimizer = dict(
    type="AdamW",
    lr=0.003,
    weight_decay=0.0001,
)

optimizer_config = dict(grad_clip=dict(max_norm=40.0, norm_type=2))
lr_config = dict(policy="CosineAnnealing", min_lr=0)
total_epochs = 5

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
resume_from = None
load_from = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA0_statedict_v3?raw=true"
