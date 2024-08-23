# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

import torch
from otx.algo.detection.heads.yolox_head import YOLOXHeadModule
from otx.algo.detection.losses import YOLOXCriterion
from otx.algo.detection.utils.assigners.sim_ota_assigner import SimOTAAssigner
from otx.algo.utils.mmengine_utils import InstanceData


class TestYOLOXCriterion:
    def test_forward(self):
        criterion = YOLOXCriterion(num_classes=4)

        s = 256
        img_metas = [
            {
                "img_shape": (s, s, 3),
                "scale_factor": 1,
            },
        ]
        train_cfg = {
            "assigner": SimOTAAssigner(center_radius=2.5),
        }
        head = YOLOXHeadModule(num_classes=4, in_channels=1, stacked_convs=1, use_depthwise=False, train_cfg=train_cfg)
        feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16]]
        cls_scores, bbox_preds, objectnesses = head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData(bboxes=torch.empty((0, 4)), labels=torch.LongTensor([]))

        raw_dict = head.loss_by_feat(cls_scores, bbox_preds, objectnesses, [gt_instances], img_metas)
        empty_gt_losses = criterion(**raw_dict)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses["loss_cls"].sum()
        empty_box_loss = empty_gt_losses["loss_bbox"].sum()
        empty_obj_loss = empty_gt_losses["loss_obj"].sum()
        assert empty_cls_loss.item() == 0, "there should be no cls loss when there are no true boxes"
        assert empty_box_loss.item() == 0, "there should be no box loss when there are no true boxes"
        assert empty_obj_loss.item() > 0, "objectness loss should be non-zero"

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        head = YOLOXHeadModule(num_classes=4, in_channels=1, stacked_convs=1, use_depthwise=True, train_cfg=train_cfg)
        head.use_l1 = True
        criterion.use_l1 = True
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
            labels=torch.LongTensor([2]),
        )

        raw_dict = head.loss_by_feat(cls_scores, bbox_preds, objectnesses, [gt_instances], img_metas)
        one_gt_losses = criterion(**raw_dict)
        onegt_cls_loss = one_gt_losses["loss_cls"].sum()
        onegt_box_loss = one_gt_losses["loss_bbox"].sum()
        onegt_obj_loss = one_gt_losses["loss_obj"].sum()
        onegt_l1_loss = one_gt_losses["loss_l1"].sum()
        assert onegt_cls_loss.item() > 0, "cls loss should be non-zero"
        assert onegt_box_loss.item() > 0, "box loss should be non-zero"
        assert onegt_obj_loss.item() > 0, "obj loss should be non-zero"
        assert onegt_l1_loss.item() > 0, "l1 loss should be non-zero"

        # Test groud truth out of bound
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[s * 4, s * 4, s * 4 + 10, s * 4 + 10]]),
            labels=torch.LongTensor([2]),
        )
        raw_dict = head.loss_by_feat(cls_scores, bbox_preds, objectnesses, [gt_instances], img_metas)
        empty_gt_losses = criterion(**raw_dict)
        # When gt_bboxes out of bound, the assign results should be empty,
        # so the cls and bbox loss should be zero.
        empty_cls_loss = empty_gt_losses["loss_cls"].sum()
        empty_box_loss = empty_gt_losses["loss_bbox"].sum()
        empty_obj_loss = empty_gt_losses["loss_obj"].sum()
        assert empty_cls_loss.item() == 0, "there should be no cls loss when gt_bboxes out of bound"
        assert empty_box_loss.item() == 0, "there should be no box loss when gt_bboxes out of bound"
        assert empty_obj_loss.item() > 0, "objectness loss should be non-zero"
