# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest
from omegaconf import DictConfig
from otx.core.data.module import OTXDataModule
from otx.core.model.entity.detection import MMDetCompatibleModel
from otx.core.utils.config import mmconfig_dict_to_dict


class TestOTXModel:
    @pytest.fixture()
    def fxt_rtmdet_tiny_model_config(self, fxt_rtmdet_tiny_config) -> DictConfig:
        return DictConfig(mmconfig_dict_to_dict(fxt_rtmdet_tiny_config.model))

    @pytest.fixture()
    def fxt_model(self, fxt_rtmdet_tiny_model_config) -> MMDetCompatibleModel:
        return MMDetCompatibleModel(config=fxt_rtmdet_tiny_model_config)

    def test_forward_train(
        self, fxt_model: MMDetCompatibleModel, fxt_datamodule: OTXDataModule,
    ) -> None:
        dataloader = fxt_datamodule.train_dataloader()
        for inputs in dataloader:
            outputs = fxt_model(inputs)
            assert isinstance(outputs, dict)
            break
