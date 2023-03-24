# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

from openvino.model_zoo._configuration import Model

from otx.core.ov.omz_wrapper import (
    download_model,
    get_model_configuration,
    get_omz_model,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_model_configuration():
    assert get_model_configuration("aa") is None
    model = get_model_configuration("mobilenet-v2-pytorch")
    assert isinstance(model, Model)
    assert getattr(model, "subdirectory") is not None
    assert getattr(model, "subdirectory_ori") is not None


@e2e_pytest_unit
def test_download_model():
    model = get_model_configuration("mobilenet-v2-pytorch")
    with tempfile.TemporaryDirectory() as tempdir:
        download_model(model, download_dir=tempdir)
        assert str(model.subdirectory) in os.listdir(tempdir)
        download_model(model, download_dir=tempdir)


@e2e_pytest_unit
def test_get_omz_model():
    with tempfile.TemporaryDirectory() as tempdir:
        model = get_omz_model("mobilenet-v2-pytorch", download_dir=tempdir, output_dir=tempdir)
        assert model is not None
        assert "model_path" in model and os.path.exists(model["model_path"])
        assert "weight_path" in model and os.path.exists(model["weight_path"])
