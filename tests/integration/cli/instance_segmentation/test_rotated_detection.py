"""Tests for rotated object detection with OTX CLI"""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import os

import pytest
import torch

from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    otx_train_testing,
    generate_model_template_testing,
)

args = {
    "--train-data-roots": "tests/assets/car_tree_bug",
    "--val-data-roots": "tests/assets/car_tree_bug",
    "--test-data-roots": "tests/assets/car_tree_bug",
    "--input": "tests/assets/car_tree_bug/images/train",
    "train_params": ["params", "--learning_parameters.num_iters", "1", "--learning_parameters.batch_size", "2"],
}

otx_dir = os.getcwd()

MULTI_GPU_UNAVAILABLE = torch.cuda.device_count() <= 1

templates = Registry("src/otx/algorithms/detection").filter(task_type="ROTATED_DETECTION").templates
templates_ids = [template.model_template_id for template in templates]


TestRotatedDetectionModelTemplates = generate_model_template_testing(templates)


# NOTE: Most of implementation parts are same with the ISeg tasks.
# So, currently just added the `test_otx_train` function to check
# Whether further modifications make Rotated detection fails or not
class TestRotatedDetectionCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_train_testing(template, tmp_dir_path, otx_dir, args)
