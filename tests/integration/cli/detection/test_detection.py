"""Tests for Class-Incremental Learning for object detection with OTX CLI"""
# Copyright (C) 2022-2023 Intel Corporation
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
    generate_model_template_testing,
)

args = {
    "--train-data-roots": "tests/assets/car_tree_bug",
    "--val-data-roots": "tests/assets/car_tree_bug",
    "--test-data-roots": "tests/assets/car_tree_bug",
    "--input": "tests/assets/car_tree_bug/images/train",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "1",
        "--learning_parameters.batch_size",
        "4",
        "--postprocessing.max_num_detections",
        "200",
    ],
}

args_semisl = {
    "--train-data-roots": "tests/assets/car_tree_bug",
    "--val-data-roots": "tests/assets/car_tree_bug",
    "--test-data-roots": "tests/assets/car_tree_bug",
    "--unlabeled-data-roots": "tests/assets/car_tree_bug",
    "--input": "tests/assets/car_tree_bug/images/train",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "1",
        "--learning_parameters.batch_size",
        "4",
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
    os.path.join("src/otx/algorithms/detection/configs", "detection", "mobilenetv2_atss", "template.yaml")
)
default_templates = [default_template]
default_templates_ids = [default_template.model_template_id]

_templates = Registry("src/otx/algorithms/detection").filter(task_type="DETECTION").templates
templates = []
for template in _templates:
    if template.name not in ["YOLOX-S", "YOLOX-X"]:
        templates.append(template)  # YOLOX-S, and YOLOX-X use same model and data pipeline config with YOLOX-L
templates_ids = [template.model_template_id for template in templates]

experimental_templates = [
    parse_model_template(
        "src/otx/algorithms/detection/configs/detection/resnet50_deformable_detr/template_experimental.yaml"
    ),
    parse_model_template("src/otx/algorithms/detection/configs/detection/resnet50_dino/template_experimental.yaml"),
    parse_model_template(
        "src/otx/algorithms/detection/configs/detection/resnet50_lite_dino/template_experimental.yaml"
    ),
]
experimental_template_ids = [template.model_template_id for template in experimental_templates]

templates_w_experimental = templates + experimental_templates
templates_ids_w_experimental = templates_ids + experimental_template_ids


TestDetectionModelTemplates = generate_model_template_testing(templates)


class TestDetectionCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_w_experimental, ids=templates_ids_w_experimental)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        _args = args.copy()
        # FIXME: remove this block once Issue#2504 resolved
        if "DINO" in template.name:
            _args["train_params"] = [
                "params",
                "--learning_parameters.num_iters",
                "1",
                "--learning_parameters.batch_size",
                "4",
                "--learning_parameters.input_size",
                "Default",
            ]
        otx_train_testing(template, tmp_dir_path, otx_dir, _args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_w_experimental, ids=templates_ids_w_experimental)
    def test_otx_resume(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection/test_resume"
        _args = args.copy()
        _resume_params = resume_params.copy()
        # FIXME: remove this block once Issue#2504 resolved
        if "DINO" in template.name:
            _args["train_params"] = [
                "params",
                "--learning_parameters.num_iters",
                "1",
                "--learning_parameters.batch_size",
                "4",
                "--learning_parameters.input_size",
                "Default",
            ]
            _resume_params.extend(["--learning_parameters.input_size", "Default"])
        otx_resume_testing(template, tmp_dir_path, otx_dir, _args)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        _args["train_params"] = _resume_params
        _args[
            "--resume-from"
        ] = f"{template_work_dir}/trained_for_resume_{template.model_template_id}/models/weights.pth"
        otx_resume_testing(template, tmp_dir_path, otx_dir, _args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_w_experimental, ids=templates_ids_w_experimental)
    @pytest.mark.parametrize("dump_features", [True, False])
    def test_otx_export(self, template, tmp_dir_path, dump_features):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_export_testing(template, tmp_dir_path, dump_features, check_ir_meta=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_w_experimental, ids=templates_ids_w_experimental)
    def test_otx_export_fp16(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_export_testing(template, tmp_dir_path, half_precision=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_w_experimental, ids=templates_ids_w_experimental)
    def test_otx_export_onnx(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_export_testing(template, tmp_dir_path, half_precision=False, is_onnx=True, check_ir_meta=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_w_experimental, ids=templates_ids_w_experimental)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_w_experimental, ids=templates_ids_w_experimental)
    @pytest.mark.parametrize("half_precision", [True, False])
    def test_otx_eval_openvino(self, template, tmp_dir_path, half_precision):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=1.0, half_precision=half_precision)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_explain_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_all_classes(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_explain_testing_all_classes(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_process_saliency_maps(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_explain_testing_process_saliency_maps(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_explain_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_all_classes_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_explain_all_classes_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_process_saliency_maps_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_explain_process_saliency_maps_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args, threshold=1.0)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_hpo(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection/test_hpo"
        otx_hpo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_multi_gpu_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection/test_multi_gpu"
        args1 = copy.deepcopy(args)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_train_semisl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection/test_semisl"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl)
        template_dir = get_template_dir(template, tmp_dir_path)
        assert os.path.exists(f"{template_dir}/semisl")

    @e2e_pytest_component
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_multi_gpu_train_semisl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection/test_multi_gpu_semisl"
        args_semisl_multigpu = copy.deepcopy(args_semisl)
        args_semisl_multigpu["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl_multigpu)
        template_dir = get_template_dir(template, tmp_dir_path)
        assert os.path.exists(f"{template_dir}/semisl")

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    @pytest.mark.parametrize("bs_adapt_type", ["Safe", "Full"])
    def test_otx_train_auto_adapt_batch_size(self, template, tmp_dir_path, bs_adapt_type):
        adapting_bs_args = copy.deepcopy(args)
        adapting_bs_args["train_params"].extend(["--learning_parameters.auto_adapt_batch_size", bs_adapt_type])
        tmp_dir_path = tmp_dir_path / f"detection_auto_adapt_{bs_adapt_type}_batch_size"
        otx_train_testing(template, tmp_dir_path, otx_dir, adapting_bs_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_train_auto_adapt_batch_size(self, template, tmp_dir_path):
        adapting_num_workers_args = copy.deepcopy(args)
        adapting_num_workers_args["train_params"].extend(["--learning_parameters.auto_num_workers", "True"])
        tmp_dir_path = tmp_dir_path / f"detection_auto_adapt_num_workers"
        otx_train_testing(template, tmp_dir_path, otx_dir, adapting_num_workers_args)
