"""Module for defining ext loader."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib


def load_ext(path, funcs):
    """A function that loads module and verifies that the specified functions are present in the module.

    :param path: str, the file path of the module to load.
    :param funcs: list of str, the names of the functions to verify in the loaded module.
    :return: the loaded module object.
    :raises: AssertionError if any of the specified functions are missing from the loaded module.
    """
    ext = importlib.import_module(path)
    for fun in funcs:
        assert hasattr(ext, fun), f"{fun} miss in module {path}"

    return ext
