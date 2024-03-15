# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch

from otx.algorithms.common.adapters.mmdeploy.utils.utils import (
    numpy_2_list,
    sync_batchnorm_2_batchnorm,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def create_model():
    class MockModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.norm = torch.nn.BatchNorm2d(3)
            self.dict_norm = torch.nn.ModuleDict(
                {
                    "0": torch.nn.BatchNorm2d(2),
                }
            )
            self.list_norm = torch.nn.ModuleList(
                [
                    torch.nn.BatchNorm2d(2),
                ]
            )

    return MockModel()


@e2e_pytest_unit
def test_convert_batchnorm():
    mock_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(create_model())
    assert isinstance(mock_model.norm, torch.nn.SyncBatchNorm)
    assert isinstance(mock_model.dict_norm["0"], torch.nn.SyncBatchNorm)
    assert isinstance(mock_model.list_norm[0], torch.nn.SyncBatchNorm)

    mock_model = sync_batchnorm_2_batchnorm(mock_model, 1)
    assert isinstance(mock_model.norm, torch.nn.BatchNorm1d)
    assert isinstance(mock_model.dict_norm["0"], torch.nn.BatchNorm1d)
    assert isinstance(mock_model.list_norm[0], torch.nn.BatchNorm1d)

    mock_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(create_model())
    assert isinstance(mock_model.norm, torch.nn.SyncBatchNorm)
    assert isinstance(mock_model.dict_norm["0"], torch.nn.SyncBatchNorm)
    assert isinstance(mock_model.list_norm[0], torch.nn.SyncBatchNorm)

    mock_model = sync_batchnorm_2_batchnorm(mock_model, 2)
    assert isinstance(mock_model.norm, torch.nn.BatchNorm2d)
    assert isinstance(mock_model.dict_norm["0"], torch.nn.BatchNorm2d)
    assert isinstance(mock_model.list_norm[0], torch.nn.BatchNorm2d)

    mock_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(create_model())
    assert isinstance(mock_model.norm, torch.nn.SyncBatchNorm)
    assert isinstance(mock_model.dict_norm["0"], torch.nn.SyncBatchNorm)
    assert isinstance(mock_model.list_norm[0], torch.nn.SyncBatchNorm)

    mock_model = sync_batchnorm_2_batchnorm(mock_model, 3)
    assert isinstance(mock_model.norm, torch.nn.BatchNorm3d)
    assert isinstance(mock_model.dict_norm["0"], torch.nn.BatchNorm3d)
    assert isinstance(mock_model.list_norm[0], torch.nn.BatchNorm3d)


@e2e_pytest_unit
def test_numpy2list():
    assert (0,) == numpy_2_list((0,))
    assert [0] == numpy_2_list([0])
    assert 0 == numpy_2_list(0)
    assert {0: 0} == numpy_2_list({0: 0})
    assert [0] == numpy_2_list(np.array([0]))
    assert 0 == numpy_2_list(np.array(0))
    assert {0: 0} == numpy_2_list({0: np.array(0)})
