# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict

import mmcv
import pytest
import torch
from mmdet.models.builder import build_head

from otx.algorithms.detection.adapters.mmdet.models.heads import *


@pytest.fixture
def fxt_head_input(
    img_size=256,
    n_bboxes=3,
    n_classes=4,
    batch_size=2,
    n_channels=64,
):
    img_metas = [
        {"img_shape": (img_size, img_size, 3), "scale_factor": 1, "pad_shape": (img_size, img_size, 3)}
        for _ in range(batch_size)
    ]

    def _gen_gt_bboxes():
        gt_bboxes = torch.rand(size=[n_bboxes, 4])
        gt_bboxes[:, :2] = img_size * 0.5 * gt_bboxes[:, :2]
        gt_bboxes[:, 2:] = img_size * (0.5 * gt_bboxes[:, 2:] + 0.5)
        return gt_bboxes.clamp(0, img_size)

    feat = [
        torch.rand(batch_size, n_channels, img_size // feat_size, img_size // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    gt_bboxes = [_gen_gt_bboxes() for _ in range(batch_size)]
    gt_labels = [torch.randint(0, n_classes, size=(n_bboxes,)) for _ in range(batch_size)]
    gt_bboxes_ignore = None
    return feat, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore


@pytest.fixture
def fxt_cfg_atss_head(n_classes=4, n_channels=64) -> Dict:
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(type="ATSSAssigner", topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        )
    )

    head_cfg = dict(
        type="CustomATSSHead",
        num_classes=n_classes,
        in_channels=n_channels,
        feat_channels=n_channels,
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        use_qfl=False,
        qfl_cfg=dict(
            type="QualityFocalLoss",
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
    )

    head_cfg["train_cfg"] = train_cfg
    return head_cfg


@pytest.fixture
def fxt_atss_head(fxt_cfg_atss_head: Dict) -> CustomATSSHead:
    return build_head(fxt_cfg_atss_head)
