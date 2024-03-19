# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from otx.core.utils.imports import get_otx_root_path


def test_get_otx_root_path(mocker):
    root_path = get_otx_root_path()
    assert isinstance(root_path, Path)
    assert str(root_path) == str(Path("./src/otx").resolve())

    with mocker.patch("importlib.import_module", return_value=None) and pytest.raises(ModuleNotFoundError):
        get_otx_root_path()
