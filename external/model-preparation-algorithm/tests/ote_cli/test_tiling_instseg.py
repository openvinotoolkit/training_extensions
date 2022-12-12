"""Tests for MPA Class-Incremental Learning for instance segmentation with OTE CLI"""
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
    ote_train_testing,
    ote_export_testing,
    nncf_optimize_testing,
    nncf_export_testing,
    nncf_eval_testing,
    nncf_eval_openvino_testing,
    pot_optimize_testing,
    pot_eval_testing,
)

args = {
<<<<<<< HEAD:external/model-preparation-algorithm/tests/ote_cli/test_segmentation.py
    "--train-ann-file": "data/segmentation/custom/annotations/training",
    "--train-data-roots": "data/segmentation/custom/images/training",
    "--val-ann-file": "data/segmentation/custom/annotations/training",
    "--val-data-roots": "data/segmentation/custom/images/training",
    "--test-ann-files": "data/segmentation/custom/annotations/training",
    "--test-data-roots": "data/segmentation/custom/images/training",
    "--input": "data/segmentation/custom/images/training",
    "train_params": [
        "params",
        "--learning_parameters.learning_rate_fixed_iters",
        "0",
        "--learning_parameters.learning_rate_warmup_iters",
        "25",
=======
    "--train-ann-file": "data/small_objects/annotations/instances_train.json",
    "--train-data-roots": "data/small_objects/images/train",
    "--val-ann-file": "data/small_objects/annotations/instances_val.json",
    "--val-data-roots": "data/small_objects/images/val",
    "--test-ann-files": "data/small_objects/annotations/instances_test.json",
    "--test-data-roots": "data/small_objects/images/test",
    "--input": "data/small_objects/images/train",
    "train_params": [
        "params",
>>>>>>> origin/releases/v0.4.0-geti1.1.0:external/model-preparation-algorithm/tests/ote_cli/test_tiling_instseg.py
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
<<<<<<< HEAD:external/model-preparation-algorithm/tests/ote_cli/test_segmentation.py
=======
        "--tiling_parameters.enable_tiling",
        "1",
        "--tiling_parameters.enable_adaptive_params",
        "1",
>>>>>>> origin/releases/v0.4.0-geti1.1.0:external/model-preparation-algorithm/tests/ote_cli/test_tiling_instseg.py
    ],
}

root = "/tmp/ote_cli/"
ote_dir = os.getcwd()

<<<<<<< HEAD:external/model-preparation-algorithm/tests/ote_cli/test_segmentation.py
TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join(
            "external/model-preparation-algorithm/configs", "segmentation", "ocr-lite-hrnet-18-mod2", "template.yaml"
        )
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("external/model-preparation-algorithm").filter(task_type="SEGMENTATION").templates
    templates_ids = [template.model_template_id for template in templates]


class TestToolsMPASegmentation:
=======
templates = Registry("external/model-preparation-algorithm").filter(task_type="INSTANCE_SEGMENTATION").templates
templates_ids = [template.model_template_id for template in templates]


class TestToolsSmallInstanceSeg:
>>>>>>> origin/releases/v0.4.0-geti1.1.0:external/model-preparation-algorithm/tests/ote_cli/test_tiling_instseg.py
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, _, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir)
<<<<<<< HEAD:external/model-preparation-algorithm/tests/ote_cli/test_segmentation.py
        print(f"algo_backend_dir: {algo_backend_dir}")
        print(f"work_dir: {work_dir}")
=======
>>>>>>> origin/releases/v0.4.0-geti1.1.0:external/model-preparation-algorithm/tests/ote_cli/test_tiling_instseg.py

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train(self, template):
        ote_train_testing(template, root, ote_dir, args)
<<<<<<< HEAD:external/model-preparation-algorithm/tests/ote_cli/test_segmentation.py
        _, template_work_dir, _ = get_some_vars(template, root)
        args1 = args.copy()
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        ote_train_testing(template, root, ote_dir, args1)
=======
>>>>>>> origin/releases/v0.4.0-geti1.1.0:external/model-preparation-algorithm/tests/ote_cli/test_tiling_instseg.py

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
    def test_ote_eval_openvino(self, template):
        ote_eval_openvino_testing(template, root, ote_dir, args, threshold=0.2)

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
<<<<<<< HEAD:external/model-preparation-algorithm/tests/ote_cli/test_segmentation.py
    def test_ote_hpo(self, template):
        ote_hpo_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
=======
>>>>>>> origin/releases/v0.4.0-geti1.1.0:external/model-preparation-algorithm/tests/ote_cli/test_tiling_instseg.py
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
    @pytest.mark.xfail(reason="CVS-98026")
    def test_nncf_eval(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, root, ote_dir, args, threshold=0.001)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.xfail(reason="CVS-98026")
    def test_nncf_eval_openvino(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template):
<<<<<<< HEAD:external/model-preparation-algorithm/tests/ote_cli/test_segmentation.py
        if template.model_template_id.startswith("ClassIncremental_Semantic_Segmentation_Lite-HRNet-"):
            pytest.skip("CVS-82482")
=======
>>>>>>> origin/releases/v0.4.0-geti1.1.0:external/model-preparation-algorithm/tests/ote_cli/test_tiling_instseg.py
        pot_optimize_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template):
<<<<<<< HEAD:external/model-preparation-algorithm/tests/ote_cli/test_segmentation.py
        if template.model_template_id.startswith("ClassIncremental_Semantic_Segmentation_Lite-HRNet-"):
            pytest.skip("CVS-82482")
=======
>>>>>>> origin/releases/v0.4.0-geti1.1.0:external/model-preparation-algorithm/tests/ote_cli/test_tiling_instseg.py
        pot_eval_testing(template, root, ote_dir, args)
