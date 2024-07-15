# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit test of custom heads of OTX Instance Segmentation tasks."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from otx.algo.instance_segmentation.heads import CustomRoIHead
from otx.algo.instance_segmentation.maskrcnn import MaskRCNNResNet50
from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity


@pytest.fixture()
def fxt_inst_seg_batch_entity() -> InstanceSegBatchDataEntity:
    return InstanceSegBatchDataEntity(
        batch_size=1,
        images=[torch.empty((1, 3, 480, 480))],
        bboxes=[torch.Tensor([[0.0, 0.0, 240, 240], [240, 240, 480, 480]])],
        labels=[torch.LongTensor([0, 1])],
        masks=[torch.zeros((2, 480, 480))],
        polygons=[[], []],
        imgs_info=[
            ImageInfo(
                img_idx=0,
                img_shape=(480, 480),
                ori_shape=(480, 480),
                ignored_labels=[],
            ),
        ],
    )


@pytest.fixture()
def fxt_inst_seg_batch_entity_with_ignored_label() -> InstanceSegBatchDataEntity:
    return InstanceSegBatchDataEntity(
        batch_size=1,
        images=[torch.empty((1, 3, 480, 480))],
        bboxes=[torch.Tensor([[0.0, 0.0, 240, 240], [240, 240, 480, 480]])],
        labels=[torch.LongTensor([0, 1])],
        masks=[torch.zeros((2, 480, 480))],
        polygons=[[], []],
        imgs_info=[
            ImageInfo(
                img_idx=0,
                img_shape=(480, 480),
                ori_shape=(480, 480),
                ignored_labels=[2],
            ),
        ],
    )


@pytest.fixture()
def fxt_instance_list() -> list[InstanceData]:
    data = InstanceData(
        bboxes=torch.Tensor([[0.0, 0.0, 240, 240], [240, 240, 480, 480]]),
        labels=torch.LongTensor([0, 0]),
        scores=torch.Tensor([1.0, 1.0]),
        masks=torch.zeros((2, 480, 480)),
    )
    return [data]


class TestClassIncrementalMixin:
    def test_ignore_label(
        self,
        mocker,
        fxt_inst_seg_batch_entity,
        fxt_inst_seg_batch_entity_with_ignored_label,
        fxt_instance_list,
    ) -> None:
        maskrcnn = MaskRCNNResNet50(3)
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
            fxt_inst_seg_batch_entity,
        )
        loss_with_ignore = maskrcnn.model.roi_head.loss(
            input_tensors,
            deepcopy(fxt_instance_list),
            fxt_inst_seg_batch_entity_with_ignored_label,
        )
        assert loss_with_ignore["loss_cls"] < loss_without_ignore["loss_cls"]
