# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit test of custom heads of OTX Instance Segmentation tasks."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from otx.algo.instance_segmentation.heads.custom_roi_head import CustomRoIHead
from otx.algo.instance_segmentation.maskrcnn import MaskRCNN


@pytest.fixture()
def fxt_data_sample() -> list[DetDataSample]:
    data_sample = DetDataSample(
        metainfo={
            "img_shape": (480, 480),
            "ori_shape": (480, 480),
            "scale_factor": (1.0, 1.0),
            "pad_shape": (480, 480),
            "ignored_labels": [],
        },
        gt_instances=InstanceData(
            bboxes=torch.Tensor([[0.0, 0.0, 240, 240], [240, 240, 480, 480]]),
            labels=torch.LongTensor([0, 1]),
        ),
    )
    return [data_sample]


@pytest.fixture()
def fxt_data_sample_with_ignored_label() -> list[DetDataSample]:
    data_sample = DetDataSample(
        metainfo={
            "img_shape": (480, 480),
            "ori_shape": (480, 480),
            "scale_factor": (1.0, 1.0),
            "pad_shape": (480, 480),
            "ignored_labels": [2],
        },
        gt_instances=InstanceData(
            bboxes=torch.Tensor([[0.0, 0.0, 240, 240], [240, 240, 480, 480]]),
            labels=torch.LongTensor([0, 1]),
        ),
    )
    return [data_sample]


@pytest.fixture()
def fxt_instance_list() -> list[InstanceData]:
    data = InstanceData(
        bboxes=torch.Tensor([[0.0, 0.0, 240, 240], [240, 240, 480, 480]]),
        labels=torch.LongTensor([0, 0]),
        scores=torch.Tensor([1.0, 1.0]),
    )
    return [data]


class TestClassIncrementalMixin:
    def test_ignore_label(
        self,
        mocker,
        fxt_data_sample,
        fxt_data_sample_with_ignored_label,
        fxt_instance_list,
    ) -> None:
        maskrcnn = MaskRCNN(3, "r50")
        input_tensors = [
            torch.randn([4, 256, 144, 256]),
            torch.randn([4, 256, 72, 128]),
            torch.randn([4, 256, 36, 64]),
            torch.randn([4, 256, 18, 32]),
            torch.randn([4, 256, 9, 16]),
        ]

        mocker.patch.object(
            CustomRoIHead,
            "mask_loss",
            return_value={"loss_mask": {"loss_mask": torch.Tensor([0.0])}},
        )
        loss_without_ignore = maskrcnn.model.roi_head.loss(
            input_tensors,
            deepcopy(fxt_instance_list),
            fxt_data_sample,
        )
        loss_with_ignore = maskrcnn.model.roi_head.loss(
            input_tensors,
            deepcopy(fxt_instance_list),
            fxt_data_sample_with_ignored_label,
        )
        assert loss_with_ignore["loss_cls"] < loss_without_ignore["loss_cls"]
