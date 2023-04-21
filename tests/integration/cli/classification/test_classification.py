"""Tests for Classification with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import glob
import os

import pytest
import torch

from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    get_template_dir,
    nncf_optimize_testing,
    otx_deploy_openvino_testing,
    otx_eval_deployment_testing,
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_explain_all_classes_openvino_testing,
    otx_explain_openvino_testing,
    otx_explain_process_saliency_maps_openvino_testing,
    otx_explain_testing,
    otx_explain_testing_all_classes,
    otx_explain_testing_process_saliency_maps,
    otx_export_testing,
    otx_hpo_testing,
    otx_resume_testing,
    otx_train_testing,
)

# Pre-train w/ 'label_0', 'label_1', 'label_2' classes
args = {
    "--train-data-roots": "tests/assets/classification_dataset",
    "--val-data-roots": "tests/assets/classification_dataset",
    "--test-data-roots": "tests/assets/classification_dataset",
    "--input": "tests/assets/classification_dataset/0",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "1",
        "--learning_parameters.batch_size",
        "4",
    ],
}

# Warmstart using data w/ 'intel', 'openvino', 'opencv' classes
args_selfsl = {
    "--train-data-roots": "tests/assets/classification_dataset",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "1",
        "--learning_parameters.batch_size",
        "4",
        "--algo_backend.train_type",
        "Selfsupervised",
    ],
}

# Training params for resume, num_iters*2
resume_params = [
    "params",
    "--learning_parameters.num_iters",
    "2",
    "--learning_parameters.batch_size",
    "4",
]

otx_dir = os.getcwd()


MULTI_GPU_UNAVAILABLE = torch.cuda.device_count() <= 1
default_template = parse_model_template(
    os.path.join(
        "otx/algorithms/classification",
        "configs",
        "efficientnet_b0_cls_incr",
        "template.yaml",
    )
)
default_templates = [default_template]
default_templates_ids = [default_template.model_template_id]

templates = Registry("otx/algorithms/classification").filter(task_type="CLASSIFICATION").templates
templates_ids = [template.model_template_id for template in templates]


class TestMultiClassClassificationCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_train_supcon(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_supcon"
        args1 = copy.deepcopy(args)
        args1["train_params"].extend(["--learning_parameters.enable_supcon", "True"])
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_resume(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_resume"
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
    @pytest.mark.parametrize("dump_features", [True, False])
    def test_otx_export(self, template, tmp_dir_path, dump_features):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_export_testing(template, tmp_dir_path, dump_features)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export_fp16(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_export_testing(template, tmp_dir_path, half_precision=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_explain_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_all_classes(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_explain_testing_all_classes(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_process_saliency_maps(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_explain_testing_process_saliency_maps(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_explain_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_all_classes_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_explain_all_classes_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_process_saliency_maps_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_explain_process_saliency_maps_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    @pytest.mark.parametrize("half_precision", [True, False])
    def test_otx_eval_openvino(self, template, tmp_dir_path, half_precision):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=1.0, half_precision=half_precision)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args, threshold=1.0)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_hpo(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_hpo"
        otx_hpo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_multi_gpu_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_multi_gpu"
        args1 = copy.deepcopy(args)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_train_semisl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_semisl"
        args_semisl = copy.deepcopy(args)
        args_semisl["--unlabeled-data-roots"] = args["--train-data-roots"]
        args_semisl["train_params"].extend(["--algo_backend.train_type", "Semisupervised"])
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl)

    @e2e_pytest_component
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_multi_gpu_train_semisl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_multi_gpu_semisl"
        args_semisl_multigpu = copy.deepcopy(args)
        args_semisl_multigpu["--unlabeled-data-roots"] = args["--train-data-roots"]
        args_semisl_multigpu["train_params"].extend(["--algo_backend.train_type", "Semisupervised"])
        args_semisl_multigpu["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl_multigpu)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_train_selfsl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_selfsl"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_selfsl)

    @e2e_pytest_component
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_multi_gpu_train_selfsl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_multi_gpu_selfsl"
        args_selfsl_multigpu = copy.deepcopy(args_selfsl)
        args_selfsl_multigpu["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_selfsl_multigpu)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_enable_noisy_lable_detection(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        new_args = copy.deepcopy(args)
        new_args["train_params"] += ["--algo_backend.enable_noisy_label_detection", "True"]
        otx_train_testing(template, tmp_dir_path, otx_dir, new_args)

        has_export_dir = False
        for root, _, _ in os.walk(tmp_dir_path):
            if "noisy_label_detection" == os.path.basename(root):
                has_export_dir = True
        assert has_export_dir

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_train_auto_decrease_batch_size(self, template, tmp_dir_path):
        decrease_bs_args = copy.deepcopy(args)
        decrease_bs_args["train_params"].extend(["--learning_parameters.auto_decrease_batch_size", "true"])
        tmp_dir_path = tmp_dir_path / "multi_class_cls_auto_decrease_batch_size"
        otx_train_testing(template, tmp_dir_path, otx_dir, decrease_bs_args)


# Multi-label training w/ 'car', 'tree', 'bug' classes
args_m = {
    "--train-data-roots": "tests/assets/datumaro_multilabel",
    "--val-data-roots": "tests/assets/datumaro_multilabel",
    "--test-data-roots": "tests/assets/datumaro_multilabel",
    "--input": "tests/assets/datumaro_multilabel/images/train",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "1",
        "--learning_parameters.batch_size",
        "4",
    ],
}


class TestMultilabelClassificationCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    @pytest.mark.parametrize("dump_features", [True, False])
    def test_otx_export(self, template, tmp_dir_path, dump_features):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_export_testing(template, tmp_dir_path, dump_features)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_export_fp16(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_export_testing(template, tmp_dir_path, half_precision=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_explain_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain_all_classes(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_explain_testing_all_classes(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain_process_saliency_maps(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_explain_testing_process_saliency_maps(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_explain_openvino_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain_all_classes_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_explain_all_classes_openvino_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain_process_saliency_maps_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_explain_process_saliency_maps_openvino_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    @pytest.mark.parametrize("half_precision", [True, False])
    def test_otx_eval_openvino(self, template, tmp_dir_path, half_precision):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args_m, threshold=1.0, half_precision=half_precision)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args_m, threshold=1.0)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args_m)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_train_semisl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls" / "test_semisl"
        args_semisl = copy.deepcopy(args_m)
        args_semisl["--unlabeled-data-roots"] = args_m["--train-data-roots"]
        args_semisl["train_params"].extend(["--algo_backend.train_type", "Semisupervised"])
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl)


args_h = {
    "--train-data-roots": "tests/assets/datumaro_h-label",
    "--val-data-roots": "tests/assets/datumaro_h-label",
    "--test-data-roots": "tests/assets/datumaro_h-label",
    "--input": "tests/assets/datumaro_h-label/images/train",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "1",
        "--learning_parameters.batch_size",
        "4",
    ],
}


class TestHierarchicalClassificationCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    @pytest.mark.parametrize("dump_features", [True, False])
    def test_otx_export(self, template, tmp_dir_path, dump_features):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_export_testing(template, tmp_dir_path, dump_features)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_export_fp16(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_export_testing(template, tmp_dir_path, half_precision=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_explain_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain_all_classes(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_explain_testing_all_classes(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain_process_saliency_maps(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_explain_testing_process_saliency_maps(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_explain_openvino_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain_all_classes_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_explain_all_classes_openvino_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_explain_process_saliency_maps_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_explain_process_saliency_maps_openvino_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    @pytest.mark.parametrize("half_precision", [True, False])
    def test_otx_eval_openvino(self, template, tmp_dir_path, half_precision):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args_h, threshold=1.0, half_precision=half_precision)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args_h)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args_h, threshold=1.0)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args_h)
