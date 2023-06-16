"""Test for otx.algorithms.mmdetection.adapters.mmdet.models.heads.custom_deformable_detr_head."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch
import pytest

from mmcv.utils import ConfigDict
from mmdet.models.builder import build_detector
from mmdet.models.dense_heads.deformable_detr_head import DeformableDETRHead

from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomDeformableDETRHead:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = ConfigDict(
            type="CustomDeformableDETRHead",
            num_query=300,
            num_classes=80,
            in_channels=2048,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=True,
            transformer=dict(
                type="DeformableDetrTransformer",
                encoder=dict(
                    type="DetrTransformerEncoder",
                    num_layers=6,
                    transformerlayers=dict(
                        type="BaseTransformerLayer",
                        attn_cfgs=dict(type="MultiScaleDeformableAttention", embed_dims=256),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=("self_attn", "norm", "ffn", "norm"),
                    ),
                ),
                decoder=dict(
                    type="DeformableDetrTransformerDecoder",
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type="DetrTransformerDecoderLayer",
                        attn_cfgs=[
                            dict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.1),
                            dict(type="MultiScaleDeformableAttention", embed_dims=256),
                        ],
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    ),
                ),
            ),
            positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_bbox=dict(type="L1Loss", loss_weight=5.0),
            loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        )
        self.head = build_detector(cfg)

    @e2e_pytest_unit
    def test_forward(self, mocker):
        def return_second_arg(a, b):
            return b

        mocker.patch.object(DeformableDETRHead, "forward", side_effect=return_second_arg)

        feats = (
            torch.randn([1, 256, 100, 167]),
            torch.randn([1, 256, 50, 84]),
            torch.randn([1, 256, 25, 42]),
            torch.randn([1, 256, 13, 21]),
        )
        img_metas = [
            {
                "filename": None,
                "ori_filename": None,
                "ori_shape": (128, 128, 3),
                "img_shape": torch.Tensor([800, 1333]),
                "pad_shape": (800, 1333, 3),
                "scale_factor": np.array([10.4140625, 6.25, 10.4140625, 6.25], dtype=np.float32),
                "flip": False,
                "flip_direction": None,
                "img_norm_cfg": {
                    "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
                    "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
                    "to_rgb": False,
                },
            }
        ]
        out = self.head(feats, img_metas)
        assert out[0].get("batch_input_shape") == (800, 1333)
        assert out[0].get("img_shape") == (800, 1333, 3)
