"""Model configuration of SegNext-B model for Self-SL Segmentation Task."""

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
    "../../../../../recipes/stages/segmentation/selfsl.py",
    "../../../../common/adapters/mmcv/configs/backbones/segnext.py",
]


model = dict(
    type="DetConB",
    pretrained="https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_b_20230227-3ab7d230.pth",
    num_classes=256,
    num_samples=16,
    downsample=8,
    input_transform="resize_concat",
    in_index=[1, 2, 3],
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 3, 12, 3],
        drop_path_rate=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
    ),
    neck=dict(
        type="SelfSLMLP",
        in_channels=960,
        hid_channels=1920,
        out_channels=256,
        norm_cfg=dict(type="BN1d", requires_grad=True),
        with_avg_pool=False,
    ),
    head=dict(
        type="DetConHead",
        predictor=dict(
            type="SelfSLMLP",
            in_channels=256,
            hid_channels=1920,
            out_channels=256,
            norm_cfg=dict(type="BN1d", requires_grad=True),
            with_avg_pool=False,
        ),
        loss_cfg=dict(type="DetConLoss", temperature=0.1),
    ),
)

optimizer = dict(paramwise_cfg=dict(custom_keys={"pos_block": dict(decay_mult=0.0), "norm": dict(decay_mult=0.0)}))
load_from = None
resume_from = None
fp16 = None
