"""Tests for MPA Class-Incremental Learning for object detection with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from otx.cli.utils.tests import (
    get_template_dir,
    nncf_eval_openvino_testing,
    nncf_eval_testing,
    nncf_export_testing,
    nncf_optimize_testing,
    otx_demo_deployment_testing,
    otx_demo_openvino_testing,
    otx_demo_testing,
    otx_deploy_openvino_testing,
    otx_eval_deployment_testing,
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_export_testing,
    otx_hpo_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component

# Fine tuning dataset
args = {
    "--train-ann-file": "../data/jester/SC_jester_3cls_12_samples_seed_2/train_list_rawframes.txt",
    "--train-data-roots": "../data/jester/SC_jester_3cls_12_samples_seed_2/rawframes_train",
    "--val-ann-file": "../data/jester/SC_jester_3cls_12_samples_seed_2/val_list_rawframes.txt",
    "--val-data-roots": "../data/jester/SC_jester_3cls_12_samples_seed_2/rawframes_val",
    "--test-ann-files": "../data/jester/SC_jester_3cls_12_samples_seed_2/val_list_rawframes.txt",
    "--test-data-roots": "../data/jester/SC_jester_3cls_12_samples_seed_2/rawframes_val",
    "--input": "../data/jester/SC_jester_3cls_12_samples_seed_2/rawframes_train",
    "train_params": ["params", "--learning_parameters.num_iters", "4", "--learning_parameters.batch_size", "4"],
}

otx_dir = os.getcwd()

TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join("otx/algorithms/action/configs/", "classification", "x3d", "template.yaml")
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("otx/algorithms").filter(task_type="ACTION_CLASSIFICATION").templates
    templates_ids = [template.model_template_id for template in templates]

@pytest.fixture(scope="session")
def tmp_dir_path():
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestToolsMPADetection:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)
