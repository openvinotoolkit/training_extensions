# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of SingleStageDetector."""

import pytest
import torch
from otx.algo.detection.base_models.single_stage_detector import SingleStageDetector
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
                self.loss = lambda x, _: {"loss": torch.sum(x)}

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.linear(x)
                x = self.relu(x)
                return self.linear2(x)

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
    def detector(self, backbone, bbox_head):
        return SingleStageDetector(backbone=backbone, bbox_head=bbox_head)

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
