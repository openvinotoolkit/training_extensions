# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of RTDETR."""

import torch
from torch import nn
from torchvision import tv_tensors

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection.rtdetr import RTDETR
from getitune.data.entity.base import BatchLoss, ImageInfo
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.types import LabelInfo


class TestRTDETR:
    def test_customize_outputs(self, mocker):
        label_info = LabelInfo(["a", "b", "c"], ["0", "1", "2"], [["a", "b", "c"]])
        mocker.patch(
            "getitune.backend.lightning.models.detection.rtdetr.RTDETR._create_model", return_value=mocker.MagicMock()
        )
        model = RTDETR(
            model_name="rtdetr_18",
            label_info=label_info,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        model.model.load_from = None
        model.train()
        outputs = {
            "loss_bbox": torch.tensor(0.5),
            "loss_vfl": torch.tensor(0.3),
            "loss_giou": torch.tensor(0.2),
        }
        inputs = SampleBatch(
            imgs_info=[
                ImageInfo(img_idx=0, img_shape=(320, 320), ori_shape=(320, 320)),
                ImageInfo(img_idx=1, img_shape=(320, 320), ori_shape=(320, 320)),
            ],
            images=torch.randn(2, 3, 320, 320),
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
        assert isinstance(result, BatchLoss)
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

        assert isinstance(result, PredictionBatch)
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

    def test_customize_inputs_handles_degenerate_empty_bboxes(self, mocker) -> None:
        """A (1, 0)-shaped empty bboxes tensor must not reach box_convert and crash."""
        label_info = LabelInfo(["a", "b", "c"], ["0", "1", "2"], [["a", "b", "c"]])
        mocker.patch(
            "getitune.backend.lightning.models.detection.rtdetr.RTDETR._create_model", return_value=mocker.MagicMock()
        )
        model = RTDETR(
            model_name="rtdetr_18",
            label_info=label_info,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        valid_bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            format="XYXY",
            canvas_size=(320, 320),
        )
        # validate_bboxes only inspects the first non-None entry, so the degenerate
        # shape must be on a later sample to reach the model's defensive guard.
        degenerate_bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.zeros((1, 0), dtype=torch.float32),
            format="XYXY",
            canvas_size=(320, 320),
        )
        entity = SampleBatch(
            images=torch.randn(2, 3, 320, 320),
            bboxes=[valid_bboxes, degenerate_bboxes],
            labels=[torch.zeros(1, dtype=torch.long), torch.zeros(0, dtype=torch.long)],
        )

        result = model._customize_inputs(entity)

        assert result["targets"][1]["boxes"].numel() == 0
