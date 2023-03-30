"""Model configuration of ResNet18 model for Segmentation Task."""

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

_base_ = [
    "../../../../recipes/stages/segmentation/incremental.py",
    "../../../common/adapters/mmcv/configs/backbones/deeplabv3plus_resnet50.py",
]

fp16 = dict(loss_scale=512.0)
model = dict(
    type='ClassIncrEncoderDecoder',
    backbone=dict(depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
        num_classes=21,
        enable_aggregator=False,
        enable_out_norm=False,
    ),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=19,
        enable_aggregator=False,
        enable_out_norm=False,))

load_from = 'open-mmlab://resnet18_v1c'
