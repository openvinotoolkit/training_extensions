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
    batch_size=8,
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
        train_cfg=train_cfg,
    )

    return head_cfg


@pytest.fixture
def fxt_cfg_ssd_head(n_classes=4, n_channels=64) -> Dict:
    head_cfg = {
        "type": "CustomSSDHead",
        "num_classes": n_classes,
        "in_channels": (n_channels, n_channels),
        "use_depthwise": True,
        "norm_cfg": {"type": "BN"},
        "act_cfg": {"type": "ReLU"},
        "init_cfg": {"type": "Xavier", "layer": "Conv2d", "distribution": "uniform"},
        "loss_balancing": False,
        "anchor_generator": {
            "type": "SSDAnchorGeneratorClustered",
            "strides": (16, 32),
            "reclustering_anchors": True,
            "widths": [
                [38.641007923271076, 92.49516032784699, 271.4234764938237, 141.53469410876247],
                [206.04136086566515, 386.6542727907841, 716.9892752215089, 453.75609561761405, 788.4629155558277],
            ],
            "heights": [
                [48.9243877087132, 147.73088476194903, 158.23569788707474, 324.14510379107367],
                [587.6216059488938, 381.60024152086544, 323.5988913027747, 702.7486097568518, 741.4865860938451],
            ],
        },
        "bbox_coder": {
            "type": "DeltaXYWHBBoxCoder",
            "target_means": (0.0, 0.0, 0.0, 0.0),
            "target_stds": (0.1, 0.1, 0.2, 0.2),
        },
        "loss_cls": {"type": "FocalLoss", "loss_weight": 1.0, "gamma": 2, "reduction": "none"},
        "train_cfg": mmcv.ConfigDict(
            {
                "assigner": {
                    "type": "MaxIoUAssigner",
                    "min_pos_iou": 0.0,
                    "ignore_iof_thr": -1,
                    "gt_max_assign_all": False,
                    "pos_iou_thr": 0.4,
                    "neg_iou_thr": 0.4,
                },
                "smoothl1_beta": 1.0,
                "allowed_border": -1,
                "pos_weight": -1,
                "neg_pos_ratio": 3,
                "debug": False,
                "use_giou": False,
                "use_focal": False,
            }
        ),
    }

    return head_cfg


@pytest.fixture
def fxt_cfg_vfnet_head(n_classes=4, n_channels=64) -> Dict:
    head_cfg = {
        "type": "CustomVFNetHead",
        "num_classes": n_classes,
        "in_channels": n_channels,
        "stacked_convs": 3,
        "feat_channels": 256,
        "strides": [8, 16, 32, 64, 128],
        "center_sampling": False,
        "dcn_on_last_conv": False,
        "use_atss": True,
        "use_vfl": True,
        "loss_cls": {
            "type": "VarifocalLoss",
            "use_sigmoid": True,
            "alpha": 0.75,
            "gamma": 2.0,
            "iou_weighted": True,
            "loss_weight": 1.0,
        },
        "loss_bbox": {"type": "GIoULoss", "loss_weight": 1.5},
        "loss_bbox_refine": {"type": "GIoULoss", "loss_weight": 2.0},
        "train_cfg": mmcv.Config(
            {
                "assigner": {"type": "ATSSAssigner", "topk": 9},
                "allowed_border": -1,
                "pos_weight": -1,
                "debug": False,
            }
        ),
    }

    return head_cfg


@pytest.fixture
def fxt_cfg_yolox_head(n_classes=4, n_channels=64):
    return {
        "type": "CustomYOLOXHead",
        "num_classes": n_classes,
        "in_channels": n_channels,
        "feat_channels": n_channels,
        "train_cfg": mmcv.Config({"assigner": {"type": "SimOTAAssigner", "center_radius": 2.5}}),
    }
