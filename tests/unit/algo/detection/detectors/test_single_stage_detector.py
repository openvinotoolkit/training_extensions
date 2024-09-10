# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of SingleStageDetector."""

import re

import pytest
import torch
from otx.algo.detection.detectors.single_stage_detector import SingleStageDetector
from otx.algo.detection.yolov9 import YOLOv9
from otx.core.data.entity.detection import DetBatchDataEntity
from otx.core.types import LabelInfo
from torch import nn


class TestSingleStageDetector:
    @pytest.fixture()
    def backbone(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @pytest.fixture()
    def bbox_head(self):
        class BboxHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 10)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(10, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.linear(x)
                x = self.relu(x)
                return self.linear2(x)

            def prepare_loss_inputs(self, x: torch.Tensor, *args, **kwargs) -> dict:
                return {"x": x}

            def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                return self.forward(x)

            def export(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                return self.forward(x)

        return BboxHead()

    @pytest.fixture()
    def batch(self):
        inputs = torch.randn(1, 3, 32, 32)
        return DetBatchDataEntity(
            batch_size=1,
            imgs_info=[LabelInfo(["a"], [["a"]])],
            images=inputs,
            bboxes=[torch.tensor([[0.5, 0.5, 0.5, 0.5]])],
            labels=[torch.tensor([0])],
        )

    @pytest.fixture()
    def criterion(self):
        class Criterion(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor, *args, **kwargs) -> dict:
                return {"loss": torch.sum(x)}

        return Criterion()

    @pytest.fixture()
    def detector(self, backbone, bbox_head, criterion):
        return SingleStageDetector(backbone=backbone, bbox_head=bbox_head, criterion=criterion)

    def test_forward(self, detector, batch):
        output = detector.forward(batch.images)
        assert isinstance(output, torch.Tensor)

    def test_loss(self, detector, batch):
        loss = detector.loss(batch)
        assert isinstance(loss, dict)
        assert "loss" in loss
        assert isinstance(loss["loss"], torch.Tensor)

    def test_predict(self, detector, batch):
        predictions = detector.predict(batch)
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == (1, 64, 16, 4)

    def test_export(self, detector, batch):
        batch_img_metas = [{"img_shape": (32, 32)}]
        output = detector.export(batch.images, batch_img_metas)
        assert isinstance(output, torch.Tensor)

    def test_extract_feat(self, detector, batch):
        features = detector.extract_feat(batch.images)
        assert isinstance(features, torch.Tensor)
        assert features.shape == (1, 64, 16, 16)

    def test_with_neck(self, detector):
        assert isinstance(detector.with_neck, bool)
        assert not detector.with_neck

    def test_with_shared_head(self, detector):
        assert isinstance(detector.with_shared_head, bool)
        assert not detector.with_shared_head

    def test_with_bbox(self, detector):
        assert isinstance(detector.with_bbox, bool)
        assert detector.with_bbox

    def test_with_mask(self, detector):
        assert isinstance(detector.with_mask, bool)
        assert not detector.with_mask


class TestYOLOSingleStageDetector:
    @pytest.fixture()
    def detector(self) -> YOLOv9:
        return YOLOv9(model_name="yolov9_s", label_info=3)

    def test_load_from_state_dict(self, detector: YOLOv9) -> None:
        state_dict = {
            "0.conv.weight": torch.randn(32, 3, 3, 3),
            "0.bn.weight": torch.randn(32),
            "0.bn.bias": torch.randn(32),
            "0.bn.running_mean": torch.randn(32),
            "0.bn.running_var": torch.randn(32),
            "0.bn.num_batches_tracked": torch.randn([]),
            "9.conv1.conv.weight": torch.randn(128, 256, 1, 1),
            "9.conv1.bn.weight": torch.randn(128),
            "9.conv1.bn.bias": torch.randn(128),
            "9.conv1.bn.running_mean": torch.randn(128),
            "9.conv1.bn.running_var": torch.randn(128),
            "9.conv1.bn.num_batches_tracked": torch.randn([]),
            "15.conv1.conv.weight": torch.randn(128, 320, 1, 1),
            "15.conv1.bn.weight": torch.randn(128),
            "15.conv1.bn.bias": torch.randn(128),
            "15.conv1.bn.running_mean": torch.randn(128),
            "15.conv1.bn.running_var": torch.randn(128),
            "15.conv1.bn.num_batches_tracked": torch.randn([]),
        }
        updated_state_dict = state_dict.copy()
        prefix = ""
        local_metadata = {}
        strict = True
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        detector.model._load_from_state_dict(
            updated_state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        backbone_len: int = len(detector.model.backbone.module)
        neck_len: int = len(detector.model.neck.module)  # type: ignore[union-attr]
        for (k, v), (updated_k, updated_v) in zip(state_dict.items(), updated_state_dict.items()):
            match = re.match(r"^(\d+)\.(.*)$", k)
            orig_idx = int(match.group(1))
            if orig_idx < backbone_len:
                assert re.match(r"backbone.module.", updated_k)
            elif orig_idx < backbone_len + neck_len:
                assert re.match(r"neck.module.", updated_k)
            else:  # for bbox_head
                assert re.match(r"bbox_head.module.", updated_k)

            assert torch.allclose(v, updated_v)
            assert v.shape == updated_v.shape
