"""Tests for Visual Prompting with OTX CLI"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import copy
import pytest

from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    get_template_dir,
    otx_eval_testing,
    otx_resume_testing,
    otx_train_testing,
)

args_polygon = {
    "--train-data-roots": "tests/assets/car_tree_bug",
    "--val-data-roots": "tests/assets/car_tree_bug",
    "--test-data-roots": "tests/assets/car_tree_bug",
    "--input": "tests/assets/car_tree_bug/images/train",
    "train_params": [
        "params",
        "--learning_parameters.trainer.max_epochs",
        "1",
        "--learning_parameters.dataset.train_batch_size",
        "2",
        "--learning_parameters.dataset.use_mask",
        "False",
    ],
}

args_mask = {
    "--train-data-roots": "tests/assets/car_tree_bug",
    "--val-data-roots": "tests/assets/car_tree_bug",
    "--test-data-roots": "tests/assets/car_tree_bug",
    "--input": "tests/assets/car_tree_bug/images/train",
    "train_params": [
        "params",
        "--learning_parameters.trainer.max_epochs",
        "1",
        "--learning_parameters.dataset.train_batch_size",
        "2",
        "--learning_parameters.dataset.use_mask",
        "True",
    ],
}

# Training params for resume, num_iters*2
resume_params = [
    "params",
    "--learning_parameters.trainer.max_epochs",
    "2",
    "--learning_parameters.dataset.train_batch_size",
    "4",
]

otx_dir = os.getcwd()


templates = Registry("otx/algorithms/visual_prompting").filter(task_type="VISUAL_PROMPTING").templates
templates_ids = [template.model_template_id for template in templates]


class TestVisualPromptingCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("args", [args_polygon, args_mask])
    def test_otx_train(self, args, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "visual_prompting"
        otx_train_testing(template, tmp_dir_path, otx_dir, args, deterministic=False)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("args", [args_polygon, args_mask])
    def test_otx_resume(self, args, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "visual_prompting/test_resume"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args)
        args1["train_params"] = resume_params
        args1[
            "--resume-from"
        ] = f"{template_work_dir}/trained_for_resume_{template.model_template_id}/models/weights.pth"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("args", [args_polygon, args_mask])
    def test_otx_eval(self, args, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "visual_prompting"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)
