# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for utility functions in OTX."""

import importlib
import inspect
from pathlib import Path


def get_otx_root_path() -> Path:
    """Return the root path of the otx module.

    Returns:
        str: The root path of the otx module.

    Raises:
        ModuleNotFoundError: If the otx module is not found.
    """
    otx_module = importlib.import_module("otx")
    if otx_module:
        file_path = inspect.getfile(otx_module)
        return Path(file_path).parent
    msg = "Cannot found otx."
    raise ModuleNotFoundError(msg)
