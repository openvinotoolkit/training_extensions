# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for segmentation model module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from otx.core.data.entity.segmentation import SegBatchPredEntity
from otx.core.model.entity.segmentation import MMSegCompatibleModel
from otx.core.model.module.segmentation import OTXSegmentationLitModule
from torchmetrics.metric import Metric


class MockMetric(torch.nn.Module):
    def update(*args, **kwargs) -> None:
        pass


class MockModel(torch.nn.Module):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __call__(self, *args, **kwargs) -> SegBatchPredEntity:
        return SegBatchPredEntity(**self.input_dict, scores=[])


class TestOTXSegmentationModule:
    @pytest.fixture()
    def fxt_model_ckpt(self) -> dict[str, torch.Tensor]:
        return {
            "model.model.backbone.1.weight": torch.randn(3, 10),
            "model.model.backbone.1.bias": torch.randn(3, 10),
            "model.model.head.1.weight": torch.randn(10, 2),
            "model.model.head.1.bias": torch.randn(10, 2),
        }

    @pytest.fixture()
    def model(self, mocker, fxt_seg_data_entity) -> OTXSegmentationLitModule:
        # define otx model
        otx_model = mocker.MagicMock(spec=MMSegCompatibleModel)
        otx_model.config = {"decode_head": {"num_classes": 2}}
        # define lightning model
        model = OTXSegmentationLitModule(otx_model, MagicMock, MagicMock, False)
        model.model.return_value = fxt_seg_data_entity[1]
        model.val_metric = mocker.MagicMock(spec=Metric)
        model.test_metric = mocker.MagicMock(spec=Metric)

        return model

    def test_validation_step(self, mocker, model, fxt_seg_data_entity) -> None:
        mocker_update_loss = mocker.patch.object(model, "_convert_pred_entity_to_compute_metric")
        model.validation_step(fxt_seg_data_entity[2], 0)
        mocker_update_loss.assert_called_once()

    def test_test_metric(self, mocker, model, fxt_seg_data_entity) -> None:
        mocker_update_loss = mocker.patch.object(model, "_convert_pred_entity_to_compute_metric")
        model.test_step(fxt_seg_data_entity[2], 0)
        mocker_update_loss.assert_called_once()

    def test_convert_pred_entity_to_compute_metric(self, model, fxt_seg_data_entity) -> None:
        pred_entity = fxt_seg_data_entity[2]
        out = model._convert_pred_entity_to_compute_metric(pred_entity, fxt_seg_data_entity[2])
        assert isinstance(out, list)
        assert "preds" in out[-1]
        assert "target" in out[-1]
        assert out[-1]["preds"].sum() == out[-1]["target"].sum()

    def test_load_from_prev_otx_ckpt(self, model, fxt_model_ckpt) -> None:
        segnext_ckpt_otx_v1 = {
            "backbone.1.weight": torch.randn(3, 10),
            "backbone.1.bias": torch.randn(3, 10),
            "head.1.weight": torch.randn(10, 2),
            "head.1.bias": torch.randn(10, 2),
            "ham.bases.1": torch.randn(3, 10),
        }
        converted_ckpt = model._load_from_prev_otx_ckpt(segnext_ckpt_otx_v1)

        assert fxt_model_ckpt.keys() == converted_ckpt.keys()
        for src_value, dst_value in zip(converted_ckpt.values(), fxt_model_ckpt.values()):
            assert src_value.shape == dst_value.shape

        litehr_ckpt_otx_v1 = {
            "backbone.1.weight": torch.randn(3, 10),
            "backbone.1.bias": torch.randn(3, 10),
            "head.1.weight": torch.randn(10, 2),
            "head.1.bias": torch.randn(10, 2),
            "decode_head.aggregator.projects.1": torch.randn(3, 10),
        }
        converted_ckpt = model._load_from_prev_otx_ckpt(litehr_ckpt_otx_v1)

        assert fxt_model_ckpt.keys() == converted_ckpt.keys()
        for src_value, dst_value in zip(converted_ckpt.values(), fxt_model_ckpt.values()):
            assert src_value.shape == dst_value.shape
