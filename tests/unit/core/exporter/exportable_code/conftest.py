# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
from otx.core.exporter.exportable_code import demo


@pytest.fixture(scope="package", autouse=True)
def fxt_fix_exportable_code_import_error():
    # exportable_code is standalone package and files in exportable_code import 'demo_package' which is same as
    # 'otx.core.exporter.exportable_code.demo.demo_package' and it makes an error as below.
    #    import demo_package  # noqa: ERA001
    # E  ModuleNotFoundError: No module named 'demo_package'
    # To avoid import error while running test, need to
    # import 'otx.core.exporter.exportable_code.demo.demo_package' and register it as 'demo_package'.
    demo_package_file = Path(demo.__file__).parent / "demo_package" / "__init__.py"
    spec = importlib.util.spec_from_file_location("demo_package", demo_package_file)
    module = importlib.util.module_from_spec(spec)

    tmp_mod_demo_package = sys.modules.get("demo_package")
    sys.modules["demo_package"] = module

    yield

    # Revert sys.modules["demo_package"] modification
    if tmp_mod_demo_package is not None:
        sys.modules["demo_package"] = tmp_mod_demo_package
    else:
        sys.modules.pop("demo_package")
