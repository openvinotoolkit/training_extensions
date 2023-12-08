# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

import hydra
import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from otx.cli.utils.hydra import configure_hydra_outputs
from otx.core.config import register_configs
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity


class MockMetric(torch.nn.Module):
    def update(*args, **kwargs) -> None:
        pass

class MockModel(torch.nn.Module):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __call__(self, *args, **kwargs) -> SegBatchPredEntity:
        return SegBatchPredEntity(**self.input_dict, scores=[])


class TestOTXSegmentationModel:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        GlobalHydra.instance().clear()
        register_configs()
        initialize(config_path="../../../../../src/otx/config", version_base="1.3", job_name="otx_train")
        overrides_list = ['+recipe=segmentation/segnext_s.yaml', 'base.output_dir=/tmp/']
        self.cfg = compose(config_name="train", overrides=overrides_list, return_hydra_config=True)
        configure_hydra_outputs(self.cfg)
        self.mmseg_model = hydra.utils.instantiate(self.cfg.model)
        img_size = (240,240)
        self.input_dict = {"batch_size": 1,
                           "images": [torch.rand(img_size)],
                           "imgs_info": [ImageInfo(0, img_size, img_size, img_size, img_size)],
                           "masks": [torch.rand(img_size)],
                           }
        self.inputs = SegBatchDataEntity(**self.input_dict)
        self.mmseg_model.model = MockModel(self.input_dict)
        self.mmseg_model.val_metric = MockMetric()
        self.mmseg_model.test_metric = MockMetric()

    def test_validation_step(self, mocker) -> None:
        mocker_update_loss = mocker.patch.object(self.mmseg_model,
                                                 "_convert_pred_entity_to_compute_metric")
        self.mmseg_model.validation_step(self.inputs, 0)
        mocker_update_loss.assert_called_once()

    def test_test_metric(self, mocker) -> None:
        mocker_update_loss = mocker.patch.object(self.mmseg_model,
                                                 "_convert_pred_entity_to_compute_metric")
        self.mmseg_model.test_step(self.inputs, 0)
        mocker_update_loss.assert_called_once()

    def test_convert_pred_entity_to_compute_metric(self) -> None:
        pred_entity = SegBatchPredEntity(**self.input_dict, scores=[])
        out = self.mmseg_model._convert_pred_entity_to_compute_metric(pred_entity, self.inputs)
        assert isinstance(out, dict)
        assert "preds" in out
        assert "target" in out
        assert out["preds"].sum() == out["target"].sum()
