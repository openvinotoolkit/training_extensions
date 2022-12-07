"""Tests for OTX CLI commands"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from otx.cli.utils.tests import otx_build_testing, otx_find_testing
from tests.test_suite.e2e_test_system import e2e_pytest_component

otx_dir = os.getcwd()


build_backbone_args = {
    "DETECTION": "torchvision.mobilenet_v3_large",
    "INSTANCE_SEGMENTATION": "torchvision.mobilenet_v3_large",
}


@pytest.fixture(scope="session")
def tmp_dir_path():
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestToolsOTXCLI:
    @e2e_pytest_component
    def test_otx_find(self):
        otx_find_testing(otx_dir)

    @e2e_pytest_component
    def test_otx_build(self, tmp_dir_path):
        otx_build_testing(tmp_dir_path, otx_dir, build_backbone_args)
