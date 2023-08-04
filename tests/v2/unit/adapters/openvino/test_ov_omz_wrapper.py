# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import tempfile
from pathlib import Path

from openvino.model_zoo._configuration import Model
from otx.v2.adapters.openvino.omz_wrapper import (
    download_model,
    get_model_configuration,
    get_omz_model,
)


def test_get_model_configuration() -> None:
    assert get_model_configuration("aa") is None
    model = get_model_configuration("mobilenet-v2-pytorch")
    assert isinstance(model, Model)
    assert model.subdirectory is not None
    assert model.subdirectory_ori is not None


def test_download_model() -> None:
    model = get_model_configuration("mobilenet-v2-pytorch")
    with tempfile.TemporaryDirectory() as tempdir:
        download_model(model, download_dir=tempdir)
        dir_list = [_path.name for _path in Path(tempdir).iterdir()]
        assert str(model.subdirectory) in dir_list
        download_model(model, download_dir=tempdir)


def test_get_omz_model() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        model = get_omz_model("mobilenet-v2-pytorch", download_dir=tempdir, output_dir=tempdir)
        assert model is not None
        assert "model_path" in model
        assert Path(model["model_path"]).exists()
        assert "weight_path" in model
        assert Path(model["weight_path"]).exists()
