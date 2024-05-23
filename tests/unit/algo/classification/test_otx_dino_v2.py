# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from otx.algo.classification.dino_v2 import DINOv2, DINOv2RegisterClassifier
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import MulticlassClsBatchPredEntity


class TestDINOv2:
    @pytest.fixture()
    def model_freeze_backbone(self) -> None:
        mock_backbone = MagicMock()
        mock_backbone.return_value = torch.randn(1, 12)

        with patch("torch.hub.load", autospec=True) as mock_load:
            mock_load.return_value = mock_backbone

            return DINOv2(
                backbone="dinov2_vits14_reg",
                freeze_backbone=True,
                head_in_channels=12,
                num_classes=2,
            )

    def test_freeze_backbone(self, model_freeze_backbone) -> None:
        for _, v in model_freeze_backbone.backbone.named_parameters():
            assert v.requires_grad is False

    def test_forward(self, model_freeze_backbone) -> None:
        rand_img = torch.randn((1, 3, 224, 224), dtype=torch.float32)
        rand_label = torch.ones((1), dtype=torch.int64)
        outputs = model_freeze_backbone(rand_img, rand_label)
        assert isinstance(outputs, torch.Tensor)


class TestDINOv2RegisterClassifier:
    @pytest.fixture()
    def otx_model(self) -> DINOv2RegisterClassifier:
        return DINOv2RegisterClassifier(label_info=1)

    def test_create_model(self, otx_model):
        assert isinstance(otx_model.model, DINOv2)

    def test_customize_inputs(self, otx_model, fxt_multiclass_cls_batch_data_entity):
        outputs = otx_model._customize_inputs(fxt_multiclass_cls_batch_data_entity)
        assert "imgs" in outputs
        assert "labels" in outputs
        assert "imgs_info" in outputs

    def test_customize_outputs(self, otx_model, fxt_multiclass_cls_batch_data_entity):
        outputs = torch.randn(2, 10)
        otx_model.training = True
        preds = otx_model._customize_outputs(outputs, fxt_multiclass_cls_batch_data_entity)
        assert isinstance(preds, OTXBatchLossEntity)

        otx_model.training = False
        preds = otx_model._customize_outputs(outputs, fxt_multiclass_cls_batch_data_entity)
        assert isinstance(preds, MulticlassClsBatchPredEntity)

    def test_predict_step(self, otx_model, fxt_multiclass_cls_batch_data_entity):
        otx_model.eval()
        outputs = otx_model.predict_step(batch=fxt_multiclass_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, MulticlassClsBatchPredEntity)
