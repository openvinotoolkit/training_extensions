"""Unit tests for CustomDINOHead."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
import torch
from mmcv.utils import ConfigDict
from mmdet.core import build_assigner
from mmdet.core.bbox.assigners import AssignResult, HungarianAssigner
from mmdet.models.builder import build_detector

from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomDINOHead:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(5)
        cfg = ConfigDict(
            dict(
                type="CustomDINOHead",
                num_query=900,
                num_classes=80,
                in_channels=2048,
                sync_cls_avg_factor=True,
                with_box_refine=True,
                as_two_stage=True,
                transformer=dict(
                    type="CustomDINOTransformer",
                    encoder=dict(
                        type="EfficientTransformerEncoder",
                        num_expansion=3,
                        enc_scale=1,
                        num_layers=6,
                        transformerlayers=[
                            dict(
                                type="EfficientTransformerLayer",
                                enc_scale=1,
                                attn_cfgs=dict(type="MultiScaleDeformableAttention", embed_dims=256, dropout=0.0),
                                feedforward_channels=2048,
                                ffn_dropout=0.0,
                                operation_order=("self_attn", "norm", "ffn", "norm"),
                            ),
                            dict(
                                type="EfficientTransformerLayer",
                                enc_scale=1,
                                small_expand=True,
                                attn_cfgs=dict(type="MultiScaleDeformableAttention", embed_dims=256, dropout=0.0),
                                ffn_cfgs=dict(
                                    type="SmallExpandFFN",
                                    embed_dims=256,
                                    feedforward_channels=1024,
                                    num_fcs=2,
                                    ffn_drop=0.0,
                                    act_cfg=dict(type="ReLU", inplace=True),
                                ),
                                feedforward_channels=2048,
                                ffn_dropout=0.0,
                                operation_order=("self_attn", "norm", "ffn"),
                            ),
                        ],
                    ),
                    decoder=dict(
                        type="DINOTransformerDecoder",
                        num_layers=6,
                        return_intermediate=True,
                        transformerlayers=dict(
                            type="DetrTransformerDecoderLayer",
                            attn_cfgs=[
                                dict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.0),
                                dict(type="MultiScaleDeformableAttention", embed_dims=256, dropout=0.0),
                            ],
                            feedforward_channels=2048,
                            ffn_dropout=0.0,
                            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                        ),
                    ),
                ),
                positional_encoding=dict(
                    type="SinePositionalEncoding", num_feats=128, normalize=True, offset=0.0, temperature=20
                ),
                loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                loss_bbox=dict(type="L1Loss", loss_weight=5.0),
                loss_iou=dict(type="GIoULoss", loss_weight=2.0),
                dn_cfg=dict(
                    label_noise_scale=0.5,
                    box_noise_scale=1.0,  # 0.4 for DN-DETR
                    group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
                ),
            ),
        )
        self.bbox_head = build_detector(cfg)

        assigner_cfg = ConfigDict(
            type="HungarianAssigner",
            cls_cost=dict(type="FocalLossCost", weight=1.0),
            reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
            iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
        )
        self.bbox_head.assigner = build_assigner(assigner_cfg)

        test_cfg = dict(max_per_img=300)
        self.bbox_head.test_cfg = test_cfg

    @e2e_pytest_unit
    def test_forward_train(self, mocker):
        inputs = [
            torch.zeros([2, 256, 92, 95]),
            torch.zeros([2, 256, 46, 48]),
            torch.zeros([2, 256, 23, 24]),
            torch.zeros([2, 256, 12, 12]),
        ]
        gt_bboxes = [
            torch.Tensor(
                [
                    [432.2500, 514.2661, 632.6323, 638.8889],
                    [361.2484, 294.9931, 558.4751, 466.9410],
                    [616.8542, 201.9204, 752.5462, 328.1207],
                    [591.6091, 386.4883, 733.6124, 571.0562],
                    [728.8790, 255.5556, 760.0000, 408.5734],
                    [713.1008, 397.5309, 760.0000, 541.0837],
                    [246.0680, 354.9383, 427.5165, 498.4911],
                    [113.5316, 361.2483, 309.1805, 517.4211],
                    [457.4950, 654.6639, 646.8326, 736.0000],
                    [132.4654, 631.0014, 187.6889, 684.6365],
                    [217.6673, 694.1015, 298.1358, 736.0000],
                    [0.0000, 583.6763, 56.7303, 672.0164],
                    [86.7088, 675.1714, 168.7551, 736.0000],
                    [173.4885, 93.0727, 253.9570, 151.4403],
                    [738.3458, 119.8903, 760.0000, 164.0603],
                    [683.1224, 522.1536, 760.0000, 736.0000],
                ]
            ),
            torch.Tensor(
                [
                    [442.0, 279.0, 544.0, 377.0],
                    [386.0, 1.0, 497.0, 108.0],
                    [288.0, 1.0, 399.0, 84.0],
                    [154.0, 1.0, 268.0, 77.0],
                    [530.0, 163.0, 625.0, 248.0],
                    [179.0, 298.0, 278.0, 398.0],
                    [275.0, 320.0, 374.0, 420.0],
                    [525.0, 394.0, 613.0, 480.0],
                    [332.0, 160.0, 463.0, 286.0],
                    [210.0, 395.0, 308.0, 480.0],
                    [141.0, 395.0, 239.0, 480.0],
                    [106.0, 225.0, 204.0, 310.0],
                    [12.0, 1.0, 148.0, 70.0],
                    [165.0, 79.0, 396.0, 247.0],
                    [483.0, 13.0, 518.0, 52.0],
                ],
            ),
        ]
        gt_labels = [
            torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2]).long(),
            torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0]).long(),
        ]
        img_metas = [
            {
                "flip_direction": "horizontal",
                "img_shape": (736, 760, 3),
                "ori_shape": (480, 640, 3),
                "img_norm_cfg": {
                    "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
                    "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
                    "to_rgb": False,
                },
                "scale_factor": np.array([1.5139443, 1.5144033, 1.5139443, 1.5144033], dtype=np.float32),
                "flip": True,
                "pad_shape": (736, 760, 3),
                "batch_input_shape": (736, 760),
            },
            {
                "flip_direction": "horizontal",
                "img_shape": (480, 640, 3),
                "ori_shape": (480, 640, 3),
                "img_norm_cfg": {
                    "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
                    "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
                    "to_rgb": False,
                },
                "scale_factor": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                "flip": True,
                "pad_shape": (480, 640, 3),
                "batch_input_shape": (736, 760),
            },
        ]

        mock_assign_result = AssignResult(
            num_gts=16,
            gt_inds=torch.randint(0, 2, (900,)),
            max_overlaps=None,
            labels=torch.zeros(900),
        )
        mocker.patch.object(HungarianAssigner, "assign", return_value=mock_assign_result)

        losses = self.bbox_head.forward_train(inputs, img_metas, gt_bboxes, gt_labels)
        assert len(losses) == 39

    @e2e_pytest_unit
    def test_simple_test_bboxes(self):
        feats = [
            torch.zeros([2, 256, 100, 134]),
            torch.zeros([2, 256, 50, 67]),
            torch.zeros([2, 256, 25, 34]),
            torch.zeros([2, 256, 13, 17]),
        ]
        img_metas = [
            {
                "ori_shape": (480, 640, 3),
                "img_shape": (800, 1067, 3),
                "pad_shape": (800, 1067, 3),
                "scale_factor": np.array([1.6671875, 1.6666666, 1.6671875, 1.6666666], dtype=np.float32),
                "flip": False,
                "flip_direction": None,
                "img_norm_cfg": {
                    "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
                    "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
                    "to_rgb": False,
                },
                "batch_input_shape": (800, 1067),
            },
            {
                "ori_shape": (480, 640, 3),
                "img_shape": (800, 1067, 3),
                "pad_shape": (800, 1067, 3),
                "scale_factor": np.array([1.6671875, 1.6666666, 1.6671875, 1.6666666], dtype=np.float32),
                "flip": False,
                "flip_direction": None,
                "img_norm_cfg": {
                    "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
                    "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
                    "to_rgb": False,
                },
                "batch_input_shape": (800, 1067),
            },
        ]
        self.bbox_head.eval()
        results = self.bbox_head.simple_test_bboxes(feats, img_metas)
        assert len(results) == 2
        assert results[0][0].shape == torch.Size([300, 5])
        assert results[0][1].shape == torch.Size([300])
