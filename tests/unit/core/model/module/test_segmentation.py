# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for segmentation model module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from otx.cli.utils.hydra import configure_hydra_outputs
from otx.core.config import register_configs
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity

if TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from otx.core.model.module.segmentation import OTXSegmentationLitModule


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
        GlobalHydra.instance().clear()
        register_configs()
        initialize(config_path="../../../../../src/otx/config", version_base="1.3", job_name="otx_train")
        overrides_list = ['+recipe=segmentation/segnext_s.yaml', 'base.output_dir=/tmp/']
        cfg = compose(config_name="train", overrides=overrides_list, return_hydra_config=True)
        configure_hydra_outputs(cfg)
        return cfg

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
        model = hydra.utils.instantiate(config.model)
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
