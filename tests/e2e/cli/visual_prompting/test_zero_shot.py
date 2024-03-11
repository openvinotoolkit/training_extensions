"""Tests for Visual Prompting with OTX CLI"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import os

import pytest

from otx.algorithms.common.utils.utils import is_xpu_available
from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_export_testing,
    otx_train_testing,
    ptq_optimize_testing,
    ptq_validate_fq_testing,
    ptq_eval_testing,
)

args = {
    "--train-data-roots": "tests/assets/car_tree_bug_zero_shot",
    "--val-data-roots": "tests/assets/car_tree_bug_zero_shot",
    "--test-data-roots": "tests/assets/car_tree_bug_zero_shot",
    "--input": "tests/assets/car_tree_bug_zero_shot/images/train",
    "train_params": [
        "params",
        "--learning_parameters.trainer.max_epochs",
        "1",
        "--learning_parameters.dataset.train_batch_size",
        "1",
        "--learning_parameters.dataset.use_mask",
        "False",
    ],
}

if is_xpu_available():
    pytest.skip("Zero shot visual prompting task is not supported on XPU", allow_module_level=True)

otx_dir = os.getcwd()

TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join("src/otx/algorithms/visual_prompting/configs", "zero_shot_sam_tiny_vit", "template.yaml")
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]

else:
    templates = [
        template
        for template in Registry("src/otx/algorithms/visual_prompting").filter(task_type="VISUAL_PROMPTING").templates
        if "Zero_Shot" in template.name
    ]
    templates_ids = [template.model_template_id for template in templates]


class TestToolsZeroShotVisualPrompting:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "zero_shot_visual_prompting"
        otx_train_testing(template, tmp_dir_path, otx_dir, args, deterministic=True)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "zero_shot_visual_prompting"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "zero_shot_visual_prompting"
        otx_export_testing(template, tmp_dir_path, False)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export_fp16(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "zero_shot_visual_prompting"
        otx_export_testing(template, tmp_dir_path, half_precision=True)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.xfail(reason="This test is failing due to unexpected performance gap.")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("half_precision", [True, False])
    def test_otx_eval_openvino(self, template, tmp_dir_path, half_precision):
        tmp_dir_path = tmp_dir_path / "zero_shot_visual_prompting"
        otx_eval_openvino_testing(
            template,
            tmp_dir_path,
            otx_dir,
            args,
            threshold=0.3,
            half_precision=half_precision,
            is_visual_prompting=True,
        )

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ptq_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "zero_shot_visual_prompting"
        ptq_optimize_testing(template, tmp_dir_path, otx_dir, args, is_visual_prompting=True)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ptq_validate_fq(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "zero_shot_visual_prompting"
        ptq_validate_fq_testing(template, tmp_dir_path, otx_dir, "visual_prompting", type(self).__name__)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.xfail(reason="This test is failing due to unexpected performance gap.")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ptq_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "zero_shot_visual_prompting"
        ptq_eval_testing(template, tmp_dir_path, otx_dir, args, is_visual_prompting=True)
