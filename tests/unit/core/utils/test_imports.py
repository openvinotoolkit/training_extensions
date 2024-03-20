# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
from pathlib import Path

import otx
import pytest
from otx.core.utils.imports import get_otx_root_path


def test_get_otx_root_path(mocker):
    root_path = get_otx_root_path()
    assert isinstance(root_path, Path)
    otx_path = inspect.getfile(otx)
    assert root_path == Path(otx_path).parent

    with mocker.patch("importlib.import_module", return_value=None) and pytest.raises(ModuleNotFoundError):
        get_otx_root_path()
