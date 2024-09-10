# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Test of YOLOXHead.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/tests/test_models/test_dense_heads/test_yolox_head.py
"""

import torch
from omegaconf import DictConfig
from otx.algo.detection.heads.yolox_head import YOLOXHeadModule
from otx.algo.detection.utils.assigners import SimOTAAssigner
from otx.algo.modules.conv_module import Conv2dModule, DepthwiseSeparableConvModule
from otx.algo.utils.mmengine_utils import InstanceData


class TestYOLOXHeadModule:
    def test_predict_by_feat(self):
        s = 256
        img_metas = [
            {
                "img_shape": (s, s, 3),
                "scale_factor": (1.0, 1.0),
            },
        ]
        test_cfg = DictConfig({"score_thr": 0.01, "nms": {"type": "nms", "iou_threshold": 0.65}})
        head = YOLOXHeadModule(num_classes=4, in_channels=1, stacked_convs=1, use_depthwise=False, test_cfg=test_cfg)
        feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16]]
        cls_scores, bbox_preds, objectnesses = head.forward(feat)
        head.predict_by_feat(cls_scores, bbox_preds, objectnesses, img_metas, cfg=test_cfg, rescale=True, with_nms=True)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            objectnesses,
            img_metas,
            cfg=test_cfg,
            rescale=False,
            with_nms=False,
        )

    def test_prepare_loss_inputs(self, mocker):
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
        assert not head.use_l1
        assert isinstance(head.multi_level_cls_convs[0][0], Conv2dModule)

        feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16]]
        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = [InstanceData(bboxes=torch.empty((0, 4)), labels=torch.LongTensor([]))]
        mocker.patch("otx.algo.detection.heads.base_head.unpack_det_entity", return_value=(gt_instances, img_metas))

        raw_dict = head.prepare_loss_inputs(x=feat, entity=mocker.MagicMock())
        for key in [
            "flatten_objectness",
            "flatten_cls_preds",
            "flatten_bbox_preds",
            "flatten_bboxes",
            "obj_targets",
            "cls_targets",
            "bbox_targets",
            "l1_targets",
            "num_total_samples",
            "num_pos",
            "pos_masks",
        ]:
            assert key in raw_dict

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        head = YOLOXHeadModule(num_classes=4, in_channels=1, stacked_convs=1, use_depthwise=True, train_cfg=train_cfg)
        assert isinstance(head.multi_level_cls_convs[0][0], DepthwiseSeparableConvModule)
