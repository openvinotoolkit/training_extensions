"""Tests for util functions related to torch."""

import torch
from unittest.mock import MagicMock

import pytest

from tests.test_suite.e2e_test_system import e2e_pytest_unit
from otx.algorithms.common.adapters.torch.utils import utils as target_module
from otx.algorithms.common.adapters.torch.utils import (
    model_from_timm,
    convert_sync_batchnorm,
    sync_batchnorm_2_batchnorm,
)


class OrdinaryModule:
    def __init__(self):
        self.module_arr = [MagicMock() for _ in range(3)]

    def modules(self):
        return self.module_arr


def get_module(is_timm: bool = False):
    module = OrdinaryModule()
    if is_timm:
        timm_sub_module = MagicMock()
        timm_sub_module.__module__ = "timm.fake"
        module.module_arr.append(timm_sub_module)

    return module


@e2e_pytest_unit
@pytest.mark.parametrize("is_timm", [True, False])
def test_model_from_timm(is_timm):
    assert model_from_timm(get_module(is_timm)) is is_timm


@e2e_pytest_unit
@pytest.mark.parametrize("is_timm", [True, False])
def test_convert_sync_batchnorm(mocker, is_timm):
    mock_timm_cvt_sycnbn = mocker.patch.object(target_module, "timm_cvt_sycnbn")
    mock_torch = mocker.patch.object(target_module, "torch")
    model = get_module(is_timm)

    convert_sync_batchnorm(model)

    if is_timm:
        mock_timm_cvt_sycnbn.assert_called_once()
    else:
        mock_torch.nn.SyncBatchNorm.convert_sync_batchnorm.assert_called_once()


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
