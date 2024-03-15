"""Test for otx.algorithms.classification.adapters.mmcls.apis.train"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from unittest import mock
from otx.algorithms.classification.adapters.mmcls.apis.train import train_model
import mmcv
import torch
from otx.algorithms.common.utils.utils import is_xpu_available


class TestTrainModel:
    @pytest.fixture
    def mock_modules(self, mocker):
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.build_dataloader", return_value=mock.MagicMock()
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.get_root_logger", return_value=mock.MagicMock()
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.build_dataloader", return_value=mock.MagicMock()
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.wrap_distributed_model",
            return_value=mock.MagicMock(),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.wrap_non_distributed_model",
            return_value=mock.MagicMock(),
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.build_optimizer", return_value=mock.MagicMock()
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.build_runner", return_value=mock.MagicMock()
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.build_dataset", return_value=mock.MagicMock()
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.build_dataloader", return_value=mock.MagicMock()
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.build_dataloader", return_value=mock.MagicMock()
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.build_dataloader", return_value=mock.MagicMock()
        )
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.DistEvalHook", return_value=mock.MagicMock()
        )
        mocker.patch("otx.algorithms.classification.adapters.mmcls.apis.train.EvalHook", return_value=mock.MagicMock())

    @pytest.fixture
    def mmcv_cfg(self):
        return mmcv.Config(
            {
                "gpu_ids": [0],
                "seed": 42,
                "data": mock.MagicMock(),
                "device": "cpu",
                "optimizer": "SGD",
                "optimizer_config": {},
                "total_epochs": 1,
                "work_dir": "test",
                "lr_config": {},
                "checkpoint_config": {},
                "log_config": {},
                "resume_from": False,
                "load_from": "",
                "workflow": "",
            }
        )

    @pytest.fixture
    def model(self):
        return mock.MagicMock()

    @pytest.fixture
    def dataset(self):
        return mock.MagicMock()

    def test_train_model_single_dataset_no_validation(self, mock_modules, mmcv_cfg, model, dataset):
        # Create mock inputs
        _ = mock_modules
        # Call the function
        train_model(model, dataset, mmcv_cfg, validate=False)

    def test_train_model_multiple_datasets_distributed_training(self, mock_modules, mmcv_cfg, model, dataset):
        # Create mock inputs
        _ = mock_modules
        # Call the function
        train_model(model, [dataset, dataset], mmcv_cfg, distributed=True, validate=True)

    def test_train_model_specific_timestamp_and_cuda_device(self, mock_modules, mmcv_cfg, model, dataset):
        # Create mock inputs
        _ = mock_modules
        timestamp = "2024-01-01"
        device = "cuda"
        mmcv_cfg.device = "cuda"
        meta = {"info": "some_info"}
        # Call the function
        train_model(model, dataset, mmcv_cfg, timestamp=timestamp, device=device, meta=meta)

    def test_train_model_xpu_device(self, mock_modules, mmcv_cfg, model, dataset, mocker):
        # Create mock inputs
        _ = mock_modules
        device = "xpu"
        mmcv_cfg.device = "xpu"
        mocker.patch("otx.algorithms.classification.adapters.mmcls.apis.train.torch")
        mocker.patch(
            "otx.algorithms.classification.adapters.mmcls.apis.train.torch.xpu.optimize",
            return_value=(mocker.MagicMock(), mocker.MagicMock()),
        )
        # Call the function
        train_model(model, dataset, mmcv_cfg, device=device)
