"""Tests for Action Classification Task with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest

from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_export_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
)

# Finetuning arguments
# TODO: Need to change sample dataset
args = {
    "--train-data-roots": "tests/assets/cvat_dataset/action_classification/train",
    "--val-data-roots": "tests/assets/cvat_dataset/action_classification/train",
    "--test-data-roots": "tests/assets/cvat_dataset/action_classification/train",
    "train_params": ["params", "--learning_parameters.num_iters", "2", "--learning_parameters.batch_size", "4"],
}

otx_dir = os.getcwd()

TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join("otx/algorithms/action/configs", "classification", "x3d", "template.yaml")
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("otx/algorithms/action").filter(task_type="ACTION_CLASSIFICATION").templates
    templates_ids = [template.model_template_id for template in templates]


class TestToolsOTXActionClassification:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "action_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "action_cls"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "action_cls"
        otx_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "action_cls"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=1.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        if template.name == "MoViNet":
            pytest.skip(reason="[CVS-106020] MoViNet fails with POT")
        tmp_dir_path = tmp_dir_path / "action_cls"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        if template.name == "MoViNet":
            pytest.skip(reason="[CVS-106020] MoViNet fails with POT")
        tmp_dir_path = tmp_dir_path / "action_cls"
        pot_eval_testing(template, tmp_dir_path, otx_dir, args)
