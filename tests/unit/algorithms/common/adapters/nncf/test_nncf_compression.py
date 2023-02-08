# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
import torch
from torch import nn

from otx.algorithms.common.adapters.nncf.compression import (
    AccuracyAwareLrUpdater,
    NNCFMetaState,
    get_uncompressed_model,
    is_checkpoint_nncf,
    is_state_nncf,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


def helper(fn, returns_with_params):
    for item in returns_with_params:
        ret = item[0]
        args = item[1]
        kwargs = item[2]

        assert ret == fn(*args, **kwargs)


class TestNNCFMetaState:
    @e2e_pytest_unit
    def test_repr(self):
        state = NNCFMetaState(None, None, None)
        assert repr(state) == "NNCFMetaState()"

        state = NNCFMetaState({"dummy": torch.tensor(1)})
        assert repr(state) == "NNCFMetaState(state_to_build='<data>')"

        state = NNCFMetaState(None, np.array(1))
        assert repr(state) == "NNCFMetaState(data_to_build='<data>')"

        state = NNCFMetaState(None, None, {"dummy": "dummy"})
        assert repr(state) == "NNCFMetaState(compression_ctrl='<data>')"


@e2e_pytest_unit
def test_is_state_nncf():
    returns_with_params = [
        (True, [{"meta": {"nncf_enable_compression": True}}], {}),
        (False, [{"meta": {"nncf_enable_compression": False}}], {}),
        (False, [{"meta": {}}], {}),
    ]

    helper(is_state_nncf, returns_with_params)


@e2e_pytest_unit
def test_is_checkpoint_nncf():
    returns_with_params = [
        (False, ["dummy_file_path"], {}),
    ]

    helper(is_checkpoint_nncf, returns_with_params)


@pytest.fixture()
def mock_model():
    class MockModule(nn.Module):
        def __init__(self):
            super().__init__()

    return MockModule()


@e2e_pytest_unit
def test_get_uncompressed_model(mock_model):
    returns_with_params = [
        (mock_model, [mock_model], {}),
    ]

    helper(get_uncompressed_model, returns_with_params)


class TestAccuracyAwareLrUpdater:
    @e2e_pytest_unit
    def test_step(self, mock_model):
        updater = AccuracyAwareLrUpdater(mock_model)
        updater.step()

    @e2e_pytest_unit
    def test_base_lrs(self, mock_model):
        updater = AccuracyAwareLrUpdater(mock_model)
        for value in np.arange(0, 0.1, 0.005):
            updater.base_lrs = value
            assert value == updater._lr_hook.base_lr
