"""Tests for Classification with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import os

import pytest
import torch

from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    get_template_dir,
    nncf_eval_openvino_testing,
    nncf_eval_testing,
    nncf_export_testing,
    nncf_optimize_testing,
    nncf_validate_fq_testing,
    otx_demo_deployment_testing,
    otx_demo_openvino_testing,
    otx_demo_testing,
    otx_deploy_openvino_testing,
    otx_eval_deployment_testing,
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_explain_openvino_testing,
    otx_explain_testing,
    otx_export_testing,
    otx_hpo_testing,
    otx_resume_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
    pot_validate_fq_testing,
)

# Pre-train w/ 'label_0', 'label_1' classes
args0 = {
    "--train-data-roots": "tests/assets/classification_dataset",
    "--val-data-roots": "tests/assets/classification_dataset",
    "--test-data-roots": "tests/assets/classification_dataset",
    "--input": "tests/assets/classification_dataset/0",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}

# Pre-train w/ 'label_0', 'label_1', 'label_2' classes
args = {
    "--train-data-roots": "tests/assets/classification_dataset_class_incremental",
    "--val-data-roots": "tests/assets/classification_dataset_class_incremental",
    "--test-data-roots": "tests/assets/classification_dataset_class_incremental",
    "--input": "tests/assets/classification_dataset/0",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}

# Training params for resume, num_iters*2
resume_params = [
    "params",
    "--learning_parameters.num_iters",
    "4",
    "--learning_parameters.batch_size",
    "4",
]

otx_dir = os.getcwd()


MULTI_GPU_UNAVAILABLE = torch.cuda.device_count() <= 1
TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join(
            "src/otx/algorithms/classification",
            "configs",
            "efficientnet_b0_cls_incr",
            "template.yaml",
        )
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("src/otx/algorithms/classification").filter(task_type="CLASSIFICATION").templates
    templates_ids = [template.model_template_id for template in templates]


class TestToolsMultiClassClassification:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_supcon(self, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Supcon for ViT template is not supported yet.")
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_supcon"
        args1 = copy.deepcopy(args)
        args1["train_params"].extend(["--learning_parameters.enable_supcon", "True"])
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, args0)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args)
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_resume(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_resume"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args0)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args0)
        args1["train_params"] = resume_params
        args1[
            "--resume-from"
        ] = f"{template_work_dir}/trained_for_resume_{template.model_template_id}/models/weights.pth"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("dump_features", [True, False])
    def test_otx_export(self, template, tmp_dir_path, dump_features):
        if template.name == "DeiT-Tiny" and dump_features:
            pytest.skip(reason="Issue#2098 ViT template does not support dump_features.")
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_export_testing(template, tmp_dir_path, dump_features)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export_fp16(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_export_testing(template, tmp_dir_path, half_precision=True)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain(self, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Issue#2098 ViT inference does not work by FeatureVectorHook.")
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_explain_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_openvino(self, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Issue#2098 ViT inference does not work by FeatureVectorHook.")
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_explain_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("half_precision", [True, False])
    def test_otx_eval_openvino(self, template, tmp_dir_path, half_precision):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=0.2, half_precision=half_precision)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_demo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_demo_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_demo_deployment_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_hpo(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_hpo"
        otx_hpo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_validate_fq(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_validate_fq_testing(template, tmp_dir_path, otx_dir, "classification", type(self).__name__)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, tmp_dir_path, otx_dir, args, threshold=0.01)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_validate_fq(self, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Fake quantization for DeiT template is not supported yet.")
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        pot_validate_fq_testing(template, tmp_dir_path, otx_dir, "classification", type(self).__name__)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        pot_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_multi_gpu_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_multi_gpu"
        args1 = copy.deepcopy(args)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)


class TestToolsMultiClassSemiSLClassification:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_semisl"
        args_semisl = copy.deepcopy(args0)
        args_semisl["--unlabeled-data-roots"] = args["--train-data-roots"]
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_semisl"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_multi_gpu_train_semisl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_multi_gpu_semisl"
        args_semisl_multigpu = copy.deepcopy(args0)
        args_semisl_multigpu["--unlabeled-data-roots"] = args["--train-data-roots"]
        args_semisl_multigpu["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl_multigpu)


# Pre-train w/ 'car', 'tree' classes
args0_m = {
    "--train-data-roots": "tests/assets/datumaro_multilabel",
    "--val-data-roots": "tests/assets/datumaro_multilabel",
    "--test-data-roots": "tests/assets/datumaro_multilabel",
    "--input": "tests/assets/datumaro_multilabel/images/train",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}

# Class-Incremental learning w/ 'car', 'tree', 'bug' classes
# TODO: Not include incremental case yet
args_m = {
    "--train-data-roots": "tests/assets/datumaro_multilabel",
    "--val-data-roots": "tests/assets/datumaro_multilabel",
    "--test-data-roots": "tests/assets/datumaro_multilabel",
    "--input": "tests/assets/datumaro_multilabel/images/train",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}


class TestToolsMultilabelClassification:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, args0_m)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args_m)
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_resume(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls/test_resume"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args0_m)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args0_m)
        args1["train_params"] = resume_params
        args1[
            "--resume-from"
        ] = f"{template_work_dir}/trained_for_resume_{template.model_template_id}/models/weights.pth"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("dump_features", [True, False])
    def test_otx_export(self, template, tmp_dir_path, dump_features):
        if template.name == "DeiT-Tiny" and dump_features:
            pytest.skip(reason="Issue#2098 ViT template does not support dump_features.")
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_export_testing(template, tmp_dir_path, dump_features)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain(self, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Issue#2098 ViT inference does not work by FeatureVectorHook.")
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_explain_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args_m, threshold=0.2)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args_m, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_hpo(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls/test_hpo"
        otx_hpo_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_validate_fq(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_validate_fq_testing(template, tmp_dir_path, otx_dir, "classification", type(self).__name__)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, tmp_dir_path, otx_dir, args_m, threshold=0.01)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_validate_fq(self, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Fake quantization for DeiT template is not supported yet.")
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        pot_validate_fq_testing(template, tmp_dir_path, otx_dir, "classification", type(self).__name__)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        pot_eval_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_multi_gpu_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls/test_multi_gpu"
        args0 = copy.deepcopy(args_m)
        args0["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args0)


args_h = {
    "--train-data-roots": "tests/assets/datumaro_h-label",
    "--val-data-roots": "tests/assets/datumaro_h-label",
    "--test-data-roots": "tests/assets/datumaro_h-label",
    "--input": "tests/assets/datumaro_h-label/images/train",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}


class TestToolsHierarchicalClassification:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_h)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args_h)
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_resume(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls/test_resume"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args_h)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args_h)
        args1["train_params"] = resume_params
        args1[
            "--resume-from"
        ] = f"{template_work_dir}/trained_for_resume_{template.model_template_id}/models/weights.pth"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("dump_features", [True, False])
    def test_otx_export(self, template, tmp_dir_path, dump_features):
        if template.name == "DeiT-Tiny" and dump_features:
            pytest.skip(reason="Issue#2098 ViT template does not support dump_features.")
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_export_testing(template, tmp_dir_path, dump_features)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain(self, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Issue#2098 ViT inference does not work by FeatureVectorHook.")
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_explain_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args_h, threshold=0.2)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args_h, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_hpo(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls/test_hpo"
        otx_hpo_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, tmp_dir_path, otx_dir, args_h, threshold=0.01)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_validate_fq(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_validate_fq_testing(template, tmp_dir_path, otx_dir, "classification", type(self).__name__)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        pot_eval_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_validate_fq(self, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Fake quantization for DeiT template is not supported yet.")
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        pot_validate_fq_testing(template, tmp_dir_path, otx_dir, "classification", type(self).__name__)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_multi_gpu_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls/test_multi_gpu"
        args1 = copy.deepcopy(args_h)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)


# Warmstart using data w/ 'intel', 'openvino', 'opencv' classes
args_selfsl = {
    "--train-data-roots": "tests/assets/classification_dataset",
    "--train-type": "Selfsupervised",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "1",
        "--learning_parameters.batch_size",
        "4",
        "--learning_parameters.learning_rate",
        "1e-07",
    ],
}


class TestToolsSelfSLClassification:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_selfsl_train(self, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Self-SL for ViT template is not supported yet.")
        tmp_dir_path_1 = tmp_dir_path / "multi_class_cls/test_selfsl"
        otx_train_testing(template, tmp_dir_path_1, otx_dir, args_selfsl)
        template_work_dir = get_template_dir(template, tmp_dir_path_1)
        assert os.path.exists(f"{template_work_dir}/selfsl")
        args1 = copy.deepcopy(args)
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
        tmp_dir_path_2 = tmp_dir_path / "multi_class_cls/test_selfsl_sl"
        otx_train_testing(template, tmp_dir_path_2, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_selfsl_eval(self, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Self-SL for ViT template is not supported yet.")
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_selfsl_sl"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_multi_gpu_train_selfsl(self, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Self-SL for ViT template is not supported yet.")
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_multi_gpu_selfsl"
        args_selfsl_multigpu = copy.deepcopy(args_selfsl)
        args_selfsl_multigpu["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_selfsl_multigpu)
        template_dir = get_template_dir(template, tmp_dir_path)
        assert os.path.exists(f"{template_dir}/selfsl")
