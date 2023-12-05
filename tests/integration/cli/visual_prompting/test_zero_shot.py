"""Tests for Zero-shot visual prompting with OTX CLI"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    otx_eval_testing,
    otx_train_testing,
)

args = {
    "--train-data-roots": "tests/assets/car_tree_bug",
    "--val-data-roots": "tests/assets/car_tree_bug",
    "--test-data-roots": "tests/assets/car_tree_bug",
    "--input": "tests/assets/car_tree_bug/images/train",
    "train_params": [
        "params",
        "--learning_parameters.trainer.max_epochs",
        "1",
    ],
}

otx_dir = os.getcwd()


templates = [
    template
    for template in Registry("src/otx/algorithms/visual_prompting", experimental=True)
    .filter(task_type="VISUAL_PROMPTING")
    .templates
    if "Zero_Shot" in template.name
]
templates_ids = [template.model_template_id for template in templates]


class TestVisualPromptingCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "zero_shot_visual_prompting"
        otx_train_testing(template, tmp_dir_path, otx_dir, args, deterministic=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "zero_shot_visual_prompting"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)
