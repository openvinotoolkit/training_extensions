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

_base_ = ["../base/supervised.py"]

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

resume_from = None
load_from = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA0_statedict_v3?raw=true"
