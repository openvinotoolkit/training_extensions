"""Model configuration of DeepLabV3Plus_ResNet18 model for Segmentation Task."""

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

norm_cfg = dict(type="BN", requires_grad=True)
fp16 = dict(loss_scale=512.0)
model = dict(
    type="ClassIncrEncoderDecoder",
    backbone=dict(depth=18),
    decode_head=dict(
        type="ASPPHead",
        in_channels=512,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=64,
        c1_channels=12,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        enable_aggregator=False,
        enable_out_norm=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        enable_aggregator=False,
        enable_out_norm=False,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    )
)

load_from = "open-mmlab://resnet18_v1c"
