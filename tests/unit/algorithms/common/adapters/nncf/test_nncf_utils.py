# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from io import BytesIO

import pytest
import torch

from otx.algorithms.common.adapters.nncf.compression import NNCFMetaState
from otx.algorithms.common.adapters.nncf.utils.utils import (
    _is_nncf_enabled,
    check_nncf_is_enabled,
    is_accuracy_aware_training_set,
    is_in_nncf_tracing,
    is_nncf_enabled,
    load_checkpoint,
    nncf_trace,
    no_nncf_trace,
    nullcontext,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.common.adapters.mmcv.nncf.test_helpers import create_model


@e2e_pytest_unit
def test_is_nncf_enabled():
    assert _is_nncf_enabled == is_nncf_enabled()


@e2e_pytest_unit
def test_nncf_is_enabled():
    if is_nncf_enabled():
        check_nncf_is_enabled()
    else:
        with pytest.raises(RuntimeError):
            check_nncf_is_enabled()


@e2e_pytest_unit
def test_load_checkpoint():
    mock_model = create_model()
    state_dict = mock_model.state_dict()
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)

    load_checkpoint(mock_model, buffer, strict=False)
    assert state_dict.keys() == mock_model.state_dict().keys()
    for k in state_dict.keys():
        assert torch.equal(state_dict[k], mock_model.state_dict()[k])

    buffer = BytesIO()
    torch.save(
        {
            "state_dict": {"dummy": "dummy"},
            "meta": {"nncf_meta": NNCFMetaState(state_to_build=state_dict, compression_ctrl={"dummy": "dummy"})},
        },
        buffer,
    )
    buffer.seek(0)
    load_checkpoint(mock_model, buffer, strict=False)
    assert state_dict.keys() == mock_model.state_dict().keys()

    state_dict["dummy"] = torch.tensor(1)
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    with pytest.raises(RuntimeError):
        load_checkpoint(mock_model, buffer, strict=True)


@e2e_pytest_unit
def test_nullcontext():
    with nullcontext():
        pass


@e2e_pytest_unit
def test_no_nncf_trace():
    with no_nncf_trace():
        pass


@e2e_pytest_unit
def test_nncf_trace():
    with nncf_trace():
        pass


@e2e_pytest_unit
def test_is_in_nncf_tracing():
    assert is_in_nncf_tracing() is False


@e2e_pytest_unit
def test_is_accuracy_aware_training_set():
    assert is_accuracy_aware_training_set({"accuracy_aware_training": True}) is True
    assert is_accuracy_aware_training_set({}) is False
    assert is_accuracy_aware_training_set({"accuracy_aware_training": {}}) is True
