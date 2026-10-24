# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
from pathlib import Path

import pytest

import getitune
from getitune.utils import get_getitune_root_path


def test_get_getitune_root_path(mocker):
    root_path = get_getitune_root_path()
    assert isinstance(root_path, Path)
    getitune_path = inspect.getfile(getitune)
    assert root_path == Path(getitune_path).parent

    with mocker.patch("importlib.import_module", return_value=None) and pytest.raises(ModuleNotFoundError):
        get_getitune_root_path()
