# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit test of custom heads of OTX Instance Segmentation tasks."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from otx.algo.common.losses import CrossEntropyLoss, CrossSigmoidFocalLoss, L1Loss
from otx.algo.common.utils.coders import DeltaXYWHBBoxCoder
from otx.algo.instance_segmentation.losses import ROICriterion
from otx.algo.instance_segmentation.maskrcnn import MaskRCNN
from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity
from torchvision import tv_tensors


@pytest.fixture()
def fxt_inst_seg_batch_entity() -> InstanceSegBatchDataEntity:
    return InstanceSegBatchDataEntity(
        batch_size=1,
        images=[torch.empty((1, 3, 480, 480))],
        bboxes=[torch.Tensor([[0.0, 0.0, 240, 240], [240, 240, 480, 480]])],
        labels=[torch.LongTensor([0, 1])],
        masks=[tv_tensors.Mask(torch.zeros((2, 480, 480)))],
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
        masks=[tv_tensors.Mask(torch.zeros((2, 480, 480)))],
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
        masks=tv_tensors.Mask(torch.zeros((2, 480, 480))),
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
        maskrcnn = MaskRCNN(3, "maskrcnn_resnet_50")
        input_tensors = [
            torch.randn([4, 256, 144, 256]),
            torch.randn([4, 256, 72, 128]),
            torch.randn([4, 256, 36, 64]),
            torch.randn([4, 256, 18, 32]),
            torch.randn([4, 256, 9, 16]),
        ]
        roi_criterion = ROICriterion(
            num_classes=3,
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            loss_bbox=L1Loss(loss_weight=1.0),
            # TODO(someone): performance of CrossSigmoidFocalLoss is worse without mmcv
            # https://github.com/openvinotoolkit/training_extensions/pull/3431
            loss_cls=CrossSigmoidFocalLoss(loss_weight=1.0, use_sigmoid=False),
            loss_mask=CrossEntropyLoss(loss_weight=1.0, use_mask=True),
        )
        (
            bbox_results,
            mask_results,
            cls_reg_targets_roi,
            mask_targets,
            pos_labels,
        ) = maskrcnn.model.roi_head.prepare_loss_inputs(
            input_tensors,
            deepcopy(fxt_instance_list),
            fxt_inst_seg_batch_entity,
        )
        labels, label_weights, bbox_targets, bbox_weights, valid_label_mask = cls_reg_targets_roi
        loss_without_ignore = roi_criterion(
            cls_score=bbox_results["cls_score"],
            bbox_pred=bbox_results["bbox_pred"],
            labels=labels,
            label_weights=label_weights,
            bbox_targets=bbox_targets,
            bbox_weights=bbox_weights,
            mask_preds=mask_results["mask_preds"],
            mask_targets=mask_targets,
            pos_labels=pos_labels,
            valid_label_mask=valid_label_mask,
        )
        (
            bbox_results,
            mask_results,
            cls_reg_targets_roi,
            mask_targets,
            pos_labels,
        ) = maskrcnn.model.roi_head.prepare_loss_inputs(
            input_tensors,
            deepcopy(fxt_instance_list),
            fxt_inst_seg_batch_entity_with_ignored_label,
        )

        labels, label_weights, bbox_targets, bbox_weights, valid_label_mask = cls_reg_targets_roi
        loss_with_ignore = roi_criterion(
            cls_score=bbox_results["cls_score"],
            bbox_pred=bbox_results["bbox_pred"],
            labels=labels,
            label_weights=label_weights,
            bbox_targets=bbox_targets,
            bbox_weights=bbox_weights,
            mask_preds=mask_results["mask_preds"],
            mask_targets=mask_targets,
            pos_labels=pos_labels,
            valid_label_mask=valid_label_mask,
        )

        assert loss_with_ignore["loss_cls"] < loss_without_ignore["loss_cls"]
