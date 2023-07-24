"""Tests for Visual Prompting with OTX CLI"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import os

import pytest

from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    get_template_dir,
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_export_testing,
    otx_resume_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
    pot_validate_fq_testing,
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
        "--learning_parameters.dataset.train_batch_size",
        "2",
        "--learning_parameters.dataset.use_mask",
        "False",
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

TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join("src/otx/algorithms/visual_prompting/configs", "sam_vit_b", "template_experimental.yaml")
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]

else:
    templates = (
        Registry("src/otx/algorithms/visual_prompting", experimental=True)
        .filter(task_type="VISUAL_PROMPTING")
        .templates
    )
    templates_ids = [template.model_template_id for template in templates]


class TestToolsVisualPrompting:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "visual_prompting"
        otx_train_testing(template, tmp_dir_path, otx_dir, args, deterministic=False)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args)
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1, deterministic=False)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_resume(self, template, tmp_dir_path):
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
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "visual_prompting"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "visual_prompting"
        otx_export_testing(template, tmp_dir_path, False)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export_fp16(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "visual_prompting"
        otx_export_testing(template, tmp_dir_path, half_precision=True)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("half_precision", [True, False])
    def test_otx_eval_openvino(self, template, tmp_dir_path, half_precision):
        tmp_dir_path = tmp_dir_path / "visual_prompting"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=0.2, half_precision=half_precision)
