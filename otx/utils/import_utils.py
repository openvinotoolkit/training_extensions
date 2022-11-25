# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib

def import_and_get_class_from_path(module_path):
    """Import and returns a class by its path in package."""

    module_name, clz_name = module_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    clz = getattr(module, clz_name)

    return 