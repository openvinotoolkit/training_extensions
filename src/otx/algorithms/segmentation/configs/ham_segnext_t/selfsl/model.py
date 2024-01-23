"""Model configuration of SegNext-T model for Self-SL Segmentation Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

_base_ = [
    "../../../../../recipes/stages/segmentation/selfsl.py",
    "../../../../common/adapters/mmcv/configs/backbones/segnext.py",
]


model = dict(
    type="DetConB",
    pretrained="https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth",
    num_classes=256,
    num_samples=16,
    downsample=8,
    input_transform="resize_concat",
    in_index=[1, 2, 3],
    neck=dict(
        type="SelfSLMLP",
        in_channels=480,
        hid_channels=960,
        out_channels=256,
        norm_cfg=dict(type="BN1d", requires_grad=True),
        with_avg_pool=False,
    ),
    head=dict(
        type="DetConHead",
        predictor=dict(
            type="SelfSLMLP",
            in_channels=256,
            hid_channels=960,
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
