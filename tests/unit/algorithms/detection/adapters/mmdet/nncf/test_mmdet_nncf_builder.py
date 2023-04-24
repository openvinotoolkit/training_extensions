# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import pytest
import numpy as np
import torch
from nncf.torch.nncf_network import NNCFNetwork

from otx.algorithms.common.adapters.nncf.compression import NNCFMetaState
from otx.algorithms.detection.adapters.mmdet.nncf.builder import build_nncf_detector
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.common.adapters.mmcv.nncf.test_helpers import (
    create_config,
    create_dataset,
    create_model,
)


@pytest.fixture(autouse=True)
def prepare_dataset():
    create_dataset(lib="mmdet")


@pytest.fixture(scope="module")
def temp_dir():
    with tempfile.TemporaryDirectory() as tempdir:
        yield tempdir


@pytest.fixture
def nncf_model_path(temp_dir):
    return os.path.join(temp_dir, "nncf_model.bin")


@pytest.fixture(scope="module")
def state_to_build():
    model = create_model(lib="mmdet")
    return model.state_dict()


@pytest.fixture
def mock_config(temp_dir, state_to_build):
    model_path = os.path.join(temp_dir, "model.bin")
    mock_config = create_config(lib="mmdet")
    torch.save(state_to_build, model_path)
    mock_config.load_from = model_path
    return mock_config


@e2e_pytest_unit
def test_build_nncf_detector(mock_config):
    with tempfile.TemporaryDirectory() as tempdir:
        mock_config.nncf_config.log_dir = tempdir
        _, model = build_nncf_detector(mock_config)

    assert isinstance(model, NNCFNetwork)
    assert len([hook for hook in mock_config.custom_hooks if hook.type == "CompressionHook"]) == 1


@e2e_pytest_unit
def test_build_nncf_detector_not_compress_postprocessing(mock_config, state_to_build, nncf_model_path):
    with tempfile.TemporaryDirectory() as tempdir:
        mock_config.nncf_config.log_dir = tempdir
        mock_config.nncf_compress_postprocessing = False
        ctrl, model = build_nncf_detector(mock_config)

    assert isinstance(model, NNCFNetwork)
    assert len([hook for hook in mock_config.custom_hooks if hook.type == "CompressionHook"]) == 1

    # save a model for next test
    torch.save(
        {
            "meta": {
                "nncf_enable_compression": True,
                "nncf_meta": NNCFMetaState(
                    data_to_build=np.zeros((50, 50, 3)),
                    compression_ctrl=ctrl.get_compression_state(),
                    state_to_build=state_to_build,
                ),
            },
            "state_dict": model.state_dict(),
        },
        nncf_model_path,
    )


@e2e_pytest_unit
def test_build_nncf_detector_with_nncf_ckpt(mock_config, nncf_model_path):
    with tempfile.TemporaryDirectory() as tempdir:
        mock_config.nncf_config.log_dir = tempdir
        _, model = build_nncf_detector(mock_config, nncf_model_path)
    assert isinstance(model, NNCFNetwork)
    assert len([hook for hook in mock_config.custom_hooks if hook.type == "CompressionHook"]) == 1
