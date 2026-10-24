# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Test of YOLOXHead.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/tests/test_models/test_dense_heads/test_yolox_head.py
"""

import torch
from omegaconf import DictConfig

from getitune.backend.lightning.models.detection.heads.yolox_head import YOLOXHeadModule
from getitune.backend.lightning.models.detection.utils.assigners import SimOTAAssigner
from getitune.backend.lightning.models.modules.conv_module import Conv2dModule, DepthwiseSeparableConvModule
from getitune.backend.lightning.models.utils.utils import InstanceData


class TestYOLOXHeadModule:
    def test_format_no_nms_output(self):
        test_cfg = DictConfig({"score_thr": 0.01, "nms": {"type": "nms", "iou_threshold": 0.65}})
        head = YOLOXHeadModule(num_classes=4, in_channels=1, stacked_convs=1, use_depthwise=False, test_cfg=test_cfg)
        bboxes = torch.rand(2, 100, 4)
        scores = torch.rand(2, 100, 4)

        dets, labels = head._format_no_nms_output(bboxes, scores)

        assert dets.shape == (2, 100, 5)
        assert labels.shape == (2, 100)
        # bbox coords preserved
        assert torch.equal(dets[..., :4], bboxes)
        # scores match max
        expected_scores, expected_labels = scores.max(dim=-1)
        assert torch.equal(dets[..., 4], expected_scores)
        assert torch.equal(labels, expected_labels)

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

    def test_export_by_feat_without_nms(self):
        s = 256
        img_metas = [{"img_shape": (s, s, 3), "scale_factor": (1.0, 1.0)}]
        test_cfg = DictConfig({"score_thr": 0.01, "nms": {"type": "nms", "iou_threshold": 0.65}})
        head = YOLOXHeadModule(num_classes=4, in_channels=1, stacked_convs=1, use_depthwise=False, test_cfg=test_cfg)
        feat = [torch.rand(1, 1, s // feat_size, s // feat_size) for feat_size in [4, 8, 16]]
        cls_scores, bbox_preds, objectnesses = head.forward(feat)

        dets, labels = head.export_by_feat(  # pyrefly: ignore[bad-unpacking]
            cls_scores,
            bbox_preds,
            objectnesses,
            img_metas,
            cfg=test_cfg,
            with_nms=False,
        )

        # dets: (batch, num_priors, 5), labels: (batch, num_priors)
        assert dets.ndim == 3
        assert dets.shape[0] == 1
        assert dets.shape[2] == 5
        assert labels.ndim == 2
        assert labels.shape == dets.shape[:2]
        # Verify scores are within valid probability range [0, 1]
        assert (dets[..., 4] >= 0).all()
        assert (dets[..., 4] <= 1).all()
        # labels in valid range
        assert (labels >= 0).all()
        assert (labels < 4).all()

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
        mocker.patch(
            "getitune.backend.lightning.models.detection.heads.base_head.unpack_det_entity",
            return_value=(gt_instances, img_metas),
        )

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
