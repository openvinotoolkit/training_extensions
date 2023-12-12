# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for segmentation model module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from torchmetrics.metric import Metric

import pytest
import torch
from omegaconf import OmegaConf
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchPredEntity
from otx.core.model.entity.segmentation import MMSegCompatibleModel
from otx.core.model.module.segmentation import OTXSegmentationLitModule

if TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig


class MockMetric(torch.nn.Module):
    def update(*args, **kwargs) -> None:
        pass

class MockModel(torch.nn.Module):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __call__(self, *args, **kwargs) -> SegBatchPredEntity:
        return SegBatchPredEntity(**self.input_dict, scores=[])


class TestOTXSegmentationModel:
    @pytest.fixture()
    def model(self, mocker, fxt_seg_data_entity) -> OTXSegmentationLitModule:
        # define otx model
        otx_model = mocker.MagicMock(spec=MMSegCompatibleModel)
        otx_model.model = mocker.MagicMock(spec=torch.nn.Module)
        otx_model.model.decode_head = mocker.MagicMock(spec=torch.nn.Module)
        otx_model.model.decode_head.num_classes = 2
        # define lightning model
        model = OTXSegmentationLitModule(otx_model, MagicMock, MagicMock, False)
        model.model.return_value = fxt_seg_data_entity[1]
        model.val_metric = mocker.MagicMock(spec=Metric)
        model.test_metric = mocker.MagicMock(spec=Metric)

        return model

    def test_validation_step(self, mocker, model, fxt_seg_data_entity) -> None:
        mocker_update_loss = mocker.patch.object(model,
                                                 "_convert_pred_entity_to_compute_metric")
        model.validation_step(fxt_seg_data_entity[2], 0)
        mocker_update_loss.assert_called_once()

    def test_test_metric(self, mocker, model, fxt_seg_data_entity) -> None:
        mocker_update_loss = mocker.patch.object(model,
                                                 "_convert_pred_entity_to_compute_metric")
        model.test_step(fxt_seg_data_entity[2], 0)
        mocker_update_loss.assert_called_once()

    def test_convert_pred_entity_to_compute_metric(self, model, fxt_seg_data_entity) -> None:
        pred_entity = fxt_seg_data_entity[2]
        out = model._convert_pred_entity_to_compute_metric(pred_entity, fxt_seg_data_entity[2])
        assert isinstance(out, dict)
        assert "preds" in out
        assert "target" in out
        assert out["preds"].sum() == out["target"].sum()
