"""Tests for OTX CLI commands"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest

from otx.cli.tools.build import SUPPORTED_TASKS
from otx.cli.utils.tests import (
    otx_build_backbone_testing,
    otx_build_task_testing,
    otx_find_testing,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component

otx_dir = os.getcwd()


build_backbone_args = [
    ("CLASSIFICATION", "torchvision.mobilenet_v3_large"),
    ("CLASSIFICATION", "mmcls.MMOVBackbone"),
    ("DETECTION", "torchvision.mobilenet_v3_large"),
    ("INSTANCE_SEGMENTATION", "torchvision.mobilenet_v3_large"),
    ("SEGMENTATION", "torchvision.mobilenet_v3_large"),
]
build_backbone_args_ids = [f"{task}_{backbone}" for task, backbone in build_backbone_args]


class TestToolsOTXCLI:
    @e2e_pytest_component
    def test_otx_find(self):
        otx_find_testing()

    @e2e_pytest_component
    @pytest.mark.parametrize("task", SUPPORTED_TASKS, ids=SUPPORTED_TASKS)
    def test_otx_workspace_build(self, tmp_dir_path, task):
        otx_build_task_testing(tmp_dir_path, task)

    @e2e_pytest_component
    @pytest.mark.parametrize("build_backbone_args", build_backbone_args, ids=build_backbone_args_ids)
    def test_otx_backbone_build(self, tmp_dir_path, build_backbone_args):
        otx_build_backbone_testing(tmp_dir_path, build_backbone_args)
