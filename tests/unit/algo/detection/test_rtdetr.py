# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of RTDETR."""

import torch
from otx.algo.detection.rtdetr import RTDETR
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.types import LabelInfo
from torch import nn


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
