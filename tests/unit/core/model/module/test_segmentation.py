# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for segmentation model module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity
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
    def config(self) -> DictConfig:
        return OmegaConf.load("src/otx/recipe/segmentation/litehrnet_18.yaml")

    @pytest.fixture()
    def input_dict(self) -> dict:
        img_size = (240,240)
        return {"batch_size": 1,
                "images": [torch.rand(img_size)],
                "imgs_info": [ImageInfo(0, img_size, img_size, img_size, img_size)],
                "masks": [torch.rand(img_size)],
                     }

    @pytest.fixture()
    def data_inputs(self, input_dict) -> SegBatchDataEntity:
        return SegBatchDataEntity(**input_dict)

    @pytest.fixture()
    def model(self, config, input_dict) -> OTXSegmentationLitModule:
        otx_model = MMSegCompatibleModel(config.model.otx_model.config)
        model = OTXSegmentationLitModule(otx_model, MagicMock, MagicMock, False)
        model.model = MockModel(input_dict)
        model.val_metric = MockMetric()
        model.test_metric = MockMetric()

        return model

    def test_validation_step(self, mocker, model, data_inputs) -> None:
        mocker_update_loss = mocker.patch.object(model,
                                                 "_convert_pred_entity_to_compute_metric")
        model.validation_step(data_inputs, 0)
        mocker_update_loss.assert_called_once()

    def test_test_metric(self, mocker, model, data_inputs) -> None:
        mocker_update_loss = mocker.patch.object(model,
                                                 "_convert_pred_entity_to_compute_metric")
        model.test_step(data_inputs, 0)
        mocker_update_loss.assert_called_once()

    def test_convert_pred_entity_to_compute_metric(self, model, data_inputs, input_dict) -> None:
        pred_entity = SegBatchPredEntity(**input_dict, scores=[])
        out = model._convert_pred_entity_to_compute_metric(pred_entity, data_inputs)
        assert isinstance(out, dict)
        assert "preds" in out
        assert "target" in out
        assert out["preds"].sum() == out["target"].sum()
