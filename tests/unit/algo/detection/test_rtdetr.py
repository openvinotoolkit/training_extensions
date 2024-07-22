# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of RTDETR."""

import pytest
import torch
import torchvision
from otx.algo.detection.backbones import PResNet
from otx.algo.detection.heads import RTDETRTransformer
from otx.algo.detection.losses import DetrCriterion
from otx.algo.detection.necks import HybridEncoder
from otx.algo.detection.rtdetr import DETR, RTDETR
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.types import LabelInfo
from torch import nn


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
        original_size = [640, 640]
        result = rt_detr_model.postprocess(outputs, original_size)
        assert isinstance(result, tuple)
        assert len(result) == 3
        scores, boxes, labels = result
        assert isinstance(scores, list)
        assert isinstance(boxes, list)
        assert isinstance(boxes[0], torchvision.tv_tensors.BoundingBoxes)
        assert boxes[0].canvas_size == original_size
        assert isinstance(labels, list)
        assert len(scores) == 2
        assert len(boxes) == 2
        assert len(labels) == 2

    def test_rt_detr_export(self, rt_detr_model, images):
        rt_detr_model.eval()
        rt_detr_model.num_top_queries = 10
        batch_img_metas = {"img_shape": (740, 740), "scale_factor": 1.0}
        result = rt_detr_model.export(images, batch_img_metas)
        assert isinstance(result, dict)
        assert "bboxes" in result
        assert "labels" in result
        assert "scores" in result
        assert result["bboxes"].shape == (2, 10, 4)
        # ensure no scaling
        assert torch.all(result["bboxes"] < 2)


class TestRTDETR:
    def test_customize_outputs(self, mocker):
        label_info = LabelInfo(["a", "b", "c"], [["a", "b", "c"]])
        mocker.patch("otx.algo.detection.rtdetr.RTDETR._build_model", return_value=mocker.MagicMock())
        model = RTDETR(label_info)
        model.model.load_from = None
        model.train()
        outputs = {
            "loss_bbox": torch.tensor(0.5),
            "loss_vfl": torch.tensor(0.3),
            "loss_giou": torch.tensor(0.2),
        }
        inputs = DetBatchDataEntity(
            batch_size=2,
            imgs_info=[mocker.MagicMock(), mocker.MagicMock()],
            images=torch.randn(2, 3, 640, 640),
            bboxes=[
                torch.tensor([[0.2739, 0.2848, 0.3239, 0.3348], [0.1652, 0.1109, 0.2152, 0.1609]]),
                torch.tensor(
                    [
                        [0.6761, 0.8174, 0.7261, 0.8674],
                        [0.1652, 0.1109, 0.2152, 0.1609],
                        [0.2848, 0.9370, 0.3348, 0.9870],
                    ],
                ),
            ],
            labels=[torch.tensor([2, 2]), torch.tensor([8, 2, 7])],
        )
        result = model._customize_outputs(outputs, inputs)
        assert isinstance(result, OTXBatchLossEntity)
        assert "loss_bbox" in result
        assert "loss_vfl" in result
        assert "loss_giou" in result
        assert result["loss_bbox"] == torch.tensor(0.5)
        assert result["loss_vfl"] == torch.tensor(0.3)
        assert result["loss_giou"] == torch.tensor(0.2)

        model.eval()
        outputs = {
            "pred_logits": torch.randn(2, 100, 10),
            "pred_boxes": torch.randn(2, 100, 4),
        }
        model.model.postprocess = lambda *_: (
            mocker.MagicMock(torch.Tensor),
            mocker.MagicMock(torch.Tensor),
            mocker.MagicMock(torch.Tensor),
        )
        result = model._customize_outputs(outputs, inputs)

        assert isinstance(result, DetBatchPredEntity)
        assert isinstance(result.scores, torch.Tensor)
        assert isinstance(result.bboxes, torch.Tensor)
        assert isinstance(result.labels, torch.Tensor)

    def test_get_optim_params(self):
        model = nn.Module()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        model.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        model.fc = nn.Linear(64, 10)

        cfg = [{"params": "^conv", "lr": 0.01, "weight_decay": 0.0}]
        params = RTDETR._get_optim_params(cfg, model)
        assert len(params) == 2

        cfg = [{"params": "^fc", "lr": 0.01, "weight_decay": 0.0}]
        params = RTDETR._get_optim_params(cfg, model)
        assert len(params) == 2
        for p1, (name, p2) in zip(params[0]["params"], model.named_parameters()):
            if "fc" in name:
                assert not torch.is_nonzero((p1.data - p2.data).sum())

        assert params[0]["lr"] == 0.01
        assert "lr" not in params[1]

        cfg = None
        params = RTDETR._get_optim_params(cfg, model)
        for p1, p2 in zip(params, model.parameters()):
            assert not torch.is_nonzero((p1.data - p2.data).sum())

        cfg = [
            {"params": "^((?!fc).)*$", "lr": 0.01, "weight_decay": 0.0},
            {"params": "^((?!conv).)*$", "lr": 0.001, "weight_decay": 0.0},
        ]
        params = RTDETR._get_optim_params(cfg, model)
        assert len(params) == 2
        for p1, p2 in zip(params[0]["params"], [p.data for name, p in model.named_parameters() if "conv" in name]):
            assert not torch.is_nonzero((p1.data - p2.data).sum())
        for p1, p2 in zip(params[1]["params"], [p.data for name, p in model.named_parameters() if "fc" in name]):
            assert not torch.is_nonzero((p1.data - p2.data).sum())
        assert params[0]["lr"] == 0.01  # conv
        assert params[1]["lr"] == 0.001  # fc
