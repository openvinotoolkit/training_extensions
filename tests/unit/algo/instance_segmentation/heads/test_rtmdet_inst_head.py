# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit test of rtmdet_ins_heads of OTX Instance Segmentation tasks."""

from __future__ import annotations

from functools import partial
from unittest.mock import Mock

import pytest
import torch
from otx.algo.common.utils.assigners import DynamicSoftLabelAssigner
from otx.algo.common.utils.coders import DistancePointBBoxCoder
from otx.algo.common.utils.prior_generators import MlvlPointGenerator
from otx.algo.common.utils.samplers import PseudoSampler
from otx.algo.instance_segmentation.heads.rtmdet_inst_head import RTMDetInstHead
from otx.algo.modules.norm import build_norm_layer
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity
from torch import nn


def set_mock_sampling_results_list(batch_size: int) -> list[Mock]:
    sampling_results_list: list[Mock] = []
    for _ in range(batch_size):
        sampling_results = Mock()
        sampling_results.pos_priors = torch.randint(0, 100, (1, 4))
        sampling_results.pos_inds = torch.randint(0, 100, (1,))
        sampling_results.pos_assigned_gt_inds = torch.randint(0, 2, (1,))
        sampling_results_list.append(sampling_results)
    return sampling_results_list


def set_mock_batch_gt_instances(batch_size: int) -> list[Mock]:
    batch_gt_instances: list[Mock] = []
    for _ in range(batch_size):
        batch_gt_instance = Mock()
        batch_gt_instance.masks = torch.zeros(3, 640, 640)
        batch_gt_instances.append(batch_gt_instance)
    return batch_gt_instances


class TestRTMDetInsHead:
    @pytest.fixture()
    def rtmdet_ins_head(self) -> RTMDetInstHead:
        train_cfg = {
            "assigner": DynamicSoftLabelAssigner(topk=13),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }

        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.5},
            "score_thr": 0.05,
            "mask_thr_binary": 0.5,
            "max_per_img": 100,
            "min_bbox_size": 0,
            "nms_pre": 300,
        }
        return RTMDetInstHead(
            num_classes=3,
            in_channels=96,
            stacked_convs=2,
            pred_kernel_size=1,
            feat_channels=96,
            normalization=partial(build_norm_layer, nn.BatchNorm2d, requires_grad=True),
            activation=partial(nn.SiLU, inplace=True),
            anchor_generator=MlvlPointGenerator(
                offset=0,
                strides=[8, 16, 32],
            ),
            bbox_coder=DistancePointBBoxCoder(),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

    def test_prepare_mask_loss_inputs(self, mocker, rtmdet_ins_head: RTMDetInstHead) -> None:
        mocker.patch.object(rtmdet_ins_head, "_mask_predict_by_feat_single", return_value=torch.randn(1, 80, 80))

        mask_feats = torch.randn(4, 8, 80, 80)
        flatten_kernels = torch.randn(4, 8400, 10)
        sampling_results_list = set_mock_sampling_results_list(4)
        batch_gt_instances = set_mock_batch_gt_instances(4)

        results = rtmdet_ins_head.prepare_mask_loss_inputs(
            mask_feats=mask_feats,
            flatten_kernels=flatten_kernels,
            sampling_results_list=sampling_results_list,
            batch_gt_instances=batch_gt_instances,
        )

        assert "batch_pos_mask_logits" in results
        assert results["batch_pos_mask_logits"].shape == (4, 160, 160)
        assert "pos_gt_masks" in results
        assert results["pos_gt_masks"].shape == (4, 160, 160)
        assert "num_pos" in results
        assert results["num_pos"] == 4

    def test_loss_mask_by_feat_without_postive(self, mocker, rtmdet_ins_head: RTMDetInstHead) -> None:
        mocker.patch.object(rtmdet_ins_head, "_mask_predict_by_feat_single", return_value=torch.randn(0, 80, 80))

        mask_feats = torch.randn(4, 8, 80, 80)
        flatten_kernels = torch.randn(4, 8400, 10)
        sampling_results_list = set_mock_sampling_results_list(4)
        batch_gt_instances = set_mock_batch_gt_instances(4)

        results = rtmdet_ins_head.prepare_mask_loss_inputs(
            mask_feats=mask_feats,
            flatten_kernels=flatten_kernels,
            sampling_results_list=sampling_results_list,
            batch_gt_instances=batch_gt_instances,
        )

        assert "zero_loss" in results
        assert results["zero_loss"] == 0
        assert "num_pos" in results
        assert results["num_pos"] == 0

    def test_prepare_loss_inputs(self, mocker, rtmdet_ins_head: RTMDetInstHead) -> None:
        mocker.patch.object(rtmdet_ins_head, "_mask_predict_by_feat_single", return_value=torch.randn(4, 80, 80))

        x = (torch.randn(2, 96, 80, 80), torch.randn(2, 96, 40, 40), torch.randn(2, 96, 20, 20))
        entity = InstanceSegBatchDataEntity(
            batch_size=2,
            images=[torch.randn(640, 640, 3), torch.randn(640, 640, 3)],
            imgs_info=[
                ImageInfo(0, img_shape=(640, 640), ori_shape=(640, 640)),
                ImageInfo(1, img_shape=(640, 640), ori_shape=(640, 640)),
            ],
            bboxes=[torch.randn(2, 4), torch.randn(3, 4)],
            labels=[torch.randint(0, 3, (2,)), torch.randint(0, 3, (3,))],
            masks=[torch.zeros(2, 640, 640), torch.zeros(3, 640, 640)],
            polygons=[[[[0, 0], [0, 1], [1, 1], [1, 0]]], [[[0, 0], [0, 1], [1, 1], [1, 0]]]],
        )

        results = rtmdet_ins_head.prepare_loss_inputs(x, entity)
        assert "cls_score" in results
        assert "bbox_pred" in results
        assert "batch_pos_mask_logits" in results
        assert "num_pos" in results
        assert "sampling_results_list" in results
        assert len(results["sampling_results_list"]) == 2
        assert len(results["labels"]) == 3
        assert results["num_pos"] == 8
