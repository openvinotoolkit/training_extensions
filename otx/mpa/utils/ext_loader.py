# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib


def load_ext(path, funcs):
    ext = importlib.import_module(path)
    for fun in funcs:
        assert hasattr(ext, fun), f"{fun} miss in module {path}"

    return ext
