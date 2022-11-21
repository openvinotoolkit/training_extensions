"""Tests for Action Classification Task with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from otx.cli.utils.tests import otx_eval_testing, otx_train_testing
from tests.test_suite.e2e_test_system import e2e_pytest_component

# Finetuning arguments
args = {
    "--train-ann-file": "data/custom_action_recognition/custom_dataset/train_list_rawframes.txt",
    "--train-data-roots": "data/custom_action_recognition/custom_dataset/rawframes",
    "--val-ann-file": "data/custom_action_recognition/custom_dataset/train_list_rawframes.txt",
    "--val-data-roots": "data/custom_action_recognition/custom_dataset/rawframes",
    "--test-ann-files": "data/custom_action_recognition/custom_dataset/train_list_rawframes.txt",
    "--test-data-roots": "data/custom_action_recognition/custom_dataset/rawframes",
    "train_params": ["params", "--learning_parameters.num_iters", "2", "--learning_parameters.batch_size", "4"],
}

otx_dir = os.getcwd()

TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join("otx/algorithms/action/configs", "classification", "movinet", "template.yaml")
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("otx/algorithms/action").filter(task_type="ACTION_CLASSIFICATION").templates
    templates_ids = [template.model_template_id for template in templates]


@pytest.fixture(scope="session")
def tmp_dir_path():
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestToolsActionCls:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)
