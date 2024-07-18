# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of RTDETRTransformer."""

import pytest
import torch
from otx.algo.detection.heads.rtdetr_decoder import RTDETRTransformer


class TestRTDETRTransformer:
    @pytest.fixture()
    def rt_detr_transformer(self):
        return RTDETRTransformer(num_classes=10, feat_channels=[128, 128, 128], num_decoder_layers=1)

    @pytest.fixture()
    def targets(self):
        return [
            {"boxes": torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]), "labels": torch.tensor([1, 0])},
        ]

    def test_rt_detr_transformer_init(self, rt_detr_transformer):
        assert isinstance(rt_detr_transformer, RTDETRTransformer)
        assert rt_detr_transformer.num_classes == 10
        assert rt_detr_transformer.aux_loss

    def test_rt_detr_transformer_forward(self, rt_detr_transformer, targets):
        feats = [torch.randn(1, 128, 60, 60), torch.randn(1, 128, 30, 30), torch.randn(1, 128, 15, 15)]
        output = rt_detr_transformer(feats, targets)
        assert isinstance(output, dict)
        assert "pred_logits" in output
        assert "pred_boxes" in output
        assert output["pred_logits"].shape == (1, rt_detr_transformer.num_queries, rt_detr_transformer.num_classes)
        assert output["pred_boxes"].shape == (1, rt_detr_transformer.num_queries, 4)
