"""Model configuration of ResNet model for Segmentation Task."""

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

load_from = 'open-mmlab://resnet50_v1c'
fp16 = dict(loss_scale=512.0)
model = dict(type='ClassIncrEncoderDecoder')