"""Semi-SL model configuration of SegNext-s model for Segmentation Task."""

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

_base_ = [
    "../../../../../recipes/stages/segmentation/semisl_poly.py",
    "../../../../common/adapters/mmcv/configs/backbones/segnext.py",
]

ham_norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="MeanTeacherSegmentor",
    orig_type="OTXEncoderDecoder",
    unsup_weight=0.1,
    proto_weight=0.1,
    semisl_start_epoch=1,
    train_cfg=dict(mix_loss=dict(enable=False, weight=0.1)),
    test_cfg=dict(mode="whole", output_scale=5.0),
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 3, 12, 3],
        drop_path_rate=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
    ),
    decode_head=dict(
        type="LightHamHead",
        input_transform="multiple_select",
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=512,
        ham_channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(MD_S=1, MD_R=16, train_steps=6, in_channels=512, eval_steps=7, inv_t=100),
    ),
    proto_head=dict(
        in_channels=512,
        channels=512,
        norm_cfg=norm_cfg,
        dropout_ratio=0.1,
        align_corners=False,
        gamma=0.999,
        num_prototype=4,
        in_proto_channels=512,
        loss_decode=dict(type="PixelPrototypeCELoss", loss_ppc_weight=0.01, loss_ppd_weight=0.001, ignore_index=255),
    )
    # model training and testing settings
)

load_from = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_b_20230227-3ab7d230.pth"
optimizer = dict(paramwise_cfg=dict(custom_keys={"pos_block": dict(decay_mult=0.0), "norm": dict(decay_mult=0.0)}))
fp16 = dict(loss_scale=512.0)
