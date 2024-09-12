# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of DETR."""

from unittest.mock import MagicMock

import pytest
import torch
import torchvision
from otx.algo.detection.backbones import PResNet
from otx.algo.detection.heads import RTDETRTransformer
from otx.algo.detection.losses import DetrCriterion
from otx.algo.detection.necks import HybridEncoder
from otx.algo.detection.rtdetr import DETR


class TestDETR:
    @pytest.fixture()
    def rt_detr_model(self):
        num_classes = 10
        backbone = PResNet(
            depth=18,
            pretrained=False,
            return_idx=[1, 2, 3],
        )
        encoder = HybridEncoder(
            in_channels=[128, 256, 512],
            dim_feedforward=1024,
            eval_spatial_size=(640, 640),
        )
        decoder = RTDETRTransformer(
            num_classes=num_classes,
            num_decoder_layers=1,
            feat_channels=[256, 256, 256],
            eval_spatial_size=(640, 640),
        )
        criterion = DetrCriterion(
            weight_dict={"loss_vfl": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
            num_classes=num_classes,
        )
        return DETR(backbone=backbone, encoder=encoder, decoder=decoder, num_classes=10, criterion=criterion)

    @pytest.fixture()
    def targets(self):
        return [
            {
                "boxes": torch.tensor([[0.2739, 0.2848, 0.3239, 0.3348], [0.1652, 0.1109, 0.2152, 0.1609]]),
                "labels": torch.tensor([2, 2]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [0.6761, 0.8174, 0.7261, 0.8674],
                        [0.1652, 0.1109, 0.2152, 0.1609],
                        [0.2848, 0.9370, 0.3348, 0.9870],
                    ],
                ),
                "labels": torch.tensor([8, 2, 7]),
            },
        ]

    @pytest.fixture()
    def images(self):
        return torch.randn(2, 3, 640, 640)

    def test_rt_detr_forward(self, rt_detr_model, images, targets):
        rt_detr_model.train()
        output = rt_detr_model(images, targets)
        assert isinstance(output, dict)
        for key in output:
            assert key.startswith("loss_")
        assert "loss_bbox" in output
        assert "loss_vfl" in output
        assert "loss_giou" in output

    def test_rt_detr_postprocess(self, rt_detr_model):
        outputs = {
            "pred_logits": torch.randn(2, 100, 10),
            "pred_boxes": torch.randn(2, 100, 4),
        }
        original_sizes = [[640, 640], [640, 640]]
        result = rt_detr_model.postprocess(outputs, original_sizes)
        assert isinstance(result, tuple)
        assert len(result) == 3
        scores, boxes, labels = result
        assert isinstance(scores, list)
        assert isinstance(boxes, list)
        assert isinstance(boxes[0], torchvision.tv_tensors.BoundingBoxes)
        assert boxes[0].canvas_size == original_sizes[0]
        assert isinstance(labels, list)
        assert len(scores) == 2
        assert len(boxes) == 2
        assert len(labels) == 2

    def test_rt_detr_export(self, rt_detr_model, images):
        rt_detr_model.eval()
        rt_detr_model.num_top_queries = 10
        batch_img_metas = [{"img_shape": (740, 740), "scale_factor": 1.0}]
        result = rt_detr_model.export(images, batch_img_metas)
        assert isinstance(result, dict)
        assert "bboxes" in result
        assert "labels" in result
        assert "scores" in result
        assert result["bboxes"].shape == (2, 10, 4)
        # ensure no scaling
        assert torch.all(result["bboxes"] < 2)

    def test_set_input_size(self):
        input_size = 1280
        model = DETR(
            backbone=MagicMock(),
            encoder=MagicMock(),
            decoder=MagicMock(),
            num_classes=10,
            input_size=input_size,
        )

        expected_multi_scale = sorted([input_size - i * 32 for i in range(-5, 6)] + [input_size] * 2)

        assert sorted(model.multi_scale) == expected_multi_scale
