"""Tests for MPA Class-Incremental Learning for image classification with OTE CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import pytest

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component
from ote_sdk.entities.model_template import parse_model_template

from ote_cli.registry import Registry
from ote_cli.utils.tests import (
    create_venv,
    get_some_vars,
    ote_demo_deployment_testing,
    ote_demo_testing,
    ote_demo_openvino_testing,
    ote_deploy_openvino_testing,
    ote_eval_deployment_testing,
    ote_eval_openvino_testing,
    ote_eval_testing,
    ote_hpo_testing,
    ote_train_testing,
    ote_export_testing,
    ote_explain_testing,
    pot_optimize_testing,
    pot_eval_testing,
    nncf_optimize_testing,
    nncf_export_testing,
    nncf_eval_testing,
    nncf_eval_openvino_testing,
)

# Pre-train w/ 'intel', 'openvino' classes
args0 = {
    "--train-ann-file": "",
    "--train-data-roots": "data/text_recognition/initial_data",
    "--val-ann-file": "",
    "--val-data-roots": "data/text_recognition/initial_data",
    "--test-ann-files": "",
    "--test-data-roots": "data/text_recognition/initial_data",
    "--input": "data/text_recognition/initial_data/intel",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}

# Pre-train w/ 'intel', 'openvino', 'opencv' classes
args = {
    "--train-ann-file": "",
    "--train-data-roots": "data/text_recognition/IL_data",
    "--val-ann-file": "",
    "--val-data-roots": "data/text_recognition/IL_data",
    "--test-ann-files": "",
    "--test-data-roots": "data/text_recognition/IL_data",
    "--input": "data/text_recognition/IL_data/intel",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}

root = "/tmp/ote_cli/"
ote_dir = os.getcwd()

TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join(
            "external/model-preparation-algorithm/configs",
            "classification",
            "efficientnet_b0_cls_incr",
            "template.yaml",
        )
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("external/model-preparation-algorithm").filter(task_type="CLASSIFICATION").templates
    templates_ids = [template.model_template_id for template in templates]


class TestToolsMPAClassification:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, _, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train(self, template):
        ote_train_testing(template, root, ote_dir, args0)
        _, template_work_dir, _ = get_some_vars(template, root)
        args1 = args.copy()
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        ote_train_testing(template, root, ote_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_export(self, template):
        ote_export_testing(template, root)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval(self, template):
        ote_eval_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval(self, template):
        ote_explain_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_openvino(self, template):
        ote_eval_openvino_testing(template, root, ote_dir, args, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo(self, template):
        ote_demo_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_openvino(self, template):
        ote_demo_openvino_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_deploy_openvino(self, template):
        ote_deploy_openvino_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_deployment(self, template):
        ote_eval_deployment_testing(template, root, ote_dir, args, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_deployment(self, template):
        ote_demo_deployment_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_hpo(self, template):
        ote_hpo_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_export(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_export_testing(template, root)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, root, ote_dir, args, threshold=0.001)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval_openvino(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template):
        pot_optimize_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template):
        pot_eval_testing(template, root, ote_dir, args)


# Pre-train w/ 'car', 'tree' classes
args0_m = {
    "--train-ann-file": "data/car_tree_bug/annotations/multilabel_car_tree.json",
    "--train-data-roots": "data/car_tree_bug/images",
    "--val-ann-file": "data/car_tree_bug/annotations/multilabel_car_tree.json",
    "--val-data-roots": "data/car_tree_bug/images",
    "--test-ann-files": "data/car_tree_bug/annotations/multilabel_car_tree.json",
    "--test-data-roots": "data/car_tree_bug/images",
    "--input": "data/car_tree_bug/images",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}

# Class-Incremental learning w/ 'car', 'tree', 'bug' classes
args_m = {
    "--train-ann-file": "data/car_tree_bug/annotations/multilabel_default.json",
    "--train-data-roots": "data/car_tree_bug/images",
    "--val-ann-file": "data/car_tree_bug/annotations/multilabel_default.json",
    "--val-data-roots": "data/car_tree_bug/images",
    "--test-ann-files": "data/car_tree_bug/annotations/multilabel_default.json",
    "--test-data-roots": "data/car_tree_bug/images",
    "--input": "data/car_tree_bug/images",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}


class TestToolsMPAMultilabelClassification:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, _, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train(self, template):
        ote_train_testing(template, root, ote_dir, args0_m)
        _, template_work_dir, _ = get_some_vars(template, root)
        args1 = args_m.copy()
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        ote_train_testing(template, root, ote_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_export(self, template):
        ote_export_testing(template, root)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval(self, template):
        ote_eval_testing(template, root, ote_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval(self, template):
        ote_explain_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_openvino(self, template):
        ote_eval_openvino_testing(template, root, ote_dir, args_m, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo(self, template):
        pytest.skip("Demo for multi-label classification is not supported now.")
        ote_demo_testing(template, root, ote_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_openvino(self, template):
        pytest.skip("Demo for multi-label classification is not supported now.")
        ote_demo_openvino_testing(template, root, ote_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_deploy_openvino(self, template):
        ote_deploy_openvino_testing(template, root, ote_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_deployment(self, template):
        ote_eval_deployment_testing(template, root, ote_dir, args_m, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_deployment(self, template):
        pytest.xfail("Demo for multi-label classification is not supported now.")
        ote_demo_deployment_testing(template, root, ote_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_hpo(self, template):
        ote_hpo_testing(template, root, ote_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, root, ote_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_export(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_export_testing(template, root)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, root, ote_dir, args_m, threshold=0.001)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval_openvino(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, root, ote_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template):
        pot_optimize_testing(template, root, ote_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template):
        pot_eval_testing(template, root, ote_dir, args_m)


# TODO: (Jihwan) Enable C-IL test without image loading via ote-cli.
args_h = {
    "--train-ann-file": "data/car_tree_bug/annotations/hierarchical_default.json",
    "--train-data-roots": "data/car_tree_bug/images",
    "--val-ann-file": "data/car_tree_bug/annotations/hierarchical_default.json",
    "--val-data-roots": "data/car_tree_bug/images",
    "--test-ann-files": "data/car_tree_bug/annotations/hierarchical_default.json",
    "--test-data-roots": "data/car_tree_bug/images",
    "--input": "data/car_tree_bug/images",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}


class TestToolsMPAHierarchicalClassification:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, _, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train(self, template):
        ote_train_testing(template, root, ote_dir, args_h)
        _, template_work_dir, _ = get_some_vars(template, root)
        args1 = args_h.copy()
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        ote_train_testing(template, root, ote_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_export(self, template):
        ote_export_testing(template, root)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval(self, template):
        ote_eval_testing(template, root, ote_dir, args_h)
    
    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval(self, template):
        ote_explain_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_openvino(self, template):
        ote_eval_openvino_testing(template, root, ote_dir, args_h, threshold=0.02)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo(self, template):
        pytest.skip("Demo for hierarchical classification is not supported now.")
        ote_demo_testing(template, root, ote_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_openvino(self, template):
        pytest.skip("Demo for hierarchical classification is not supported now.")
        ote_demo_openvino_testing(template, root, ote_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_deploy_openvino(self, template):
        ote_deploy_openvino_testing(template, root, ote_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_deployment(self, template):
        ote_eval_deployment_testing(template, root, ote_dir, args_h, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_deployment(self, template):
        pytest.skip("Demo for hierarchical classification is not supported now.")
        ote_demo_deployment_testing(template, root, ote_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_hpo(self, template):
        ote_hpo_testing(template, root, ote_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, root, ote_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_export(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_export_testing(template, root)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, root, ote_dir, args_h, threshold=0.001)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval_openvino(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, root, ote_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template):
        pot_optimize_testing(template, root, ote_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template):
        pot_eval_testing(template, root, ote_dir, args_h)
