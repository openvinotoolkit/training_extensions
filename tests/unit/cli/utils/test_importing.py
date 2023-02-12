# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import importlib
import inspect

import pytest
from mmcv.utils import Registry

from otx.cli.utils.importing import (
    get_backbone_list,
    get_backbone_registry,
    get_impl_class,
    get_otx_root_path,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_impl_class():
    impl_path = "otx.algorithms.common.adapters.mmcv.models.OTXMobileNetV3"
    task_impl_class = get_impl_class(impl_path)
    assert task_impl_class.__name__ == "OTXMobileNetV3"


@e2e_pytest_unit
@pytest.mark.parametrize("backend", ["otx", "pytorchcv"])
def test_get_backbone_list(backend):
    available_backbones = get_backbone_list(backend)
    assert isinstance(available_backbones, dict)


@e2e_pytest_unit
def test_get_backbone_list_for_unsupported_backend():
    backend = "invalid"
    with pytest.raises(ValueError):
        get_backbone_list(backend)


@e2e_pytest_unit
def test_get_backbone_registry():
    backend = "otx"
    mm_registry, custom_imports = get_backbone_registry(backend)
    assert custom_imports == ["otx.algorithms.common.adapters.mmcv.models"]
    assert isinstance(mm_registry, Registry)


@e2e_pytest_unit
def test_get_backbone_registry_for_unsupported_backend():
    backend = "invalid"
    with pytest.raises(ValueError):
        get_backbone_registry(backend)


@e2e_pytest_unit
def test_get_otx_root_path(mocker):
    mocker.patch.object(importlib, "import_module", return_value=mocker.MagicMock())
    mocker.patch.object(inspect, "getfile", return_value="otx/__init__.py")
    otx_root_path = get_otx_root_path()
    assert otx_root_path == "otx"
