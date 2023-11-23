"""Tests for Class-Incremental Learning for object detection with OTX CLI"""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import os
from pathlib import Path

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
    otx_explain_openvino_testing,
    otx_explain_testing,
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
        "2",
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
        "2",
        "--postprocessing.max_num_detections",
        "200",
    ],
}

# Training params for resume, num_iters*2
resume_params = [
    "params",
    "--learning_parameters.num_iters",
    "2",
    "--learning_parameters.batch_size",
    "2",
    "--postprocessing.max_num_detections",
    "200",
]

otx_dir = os.getcwd()

iseg_config_root = Path("src/otx/algorithms/detection/configs/instance_segmentation")

MULTI_GPU_UNAVAILABLE = torch.cuda.device_count() <= 1
default_template = parse_model_template(iseg_config_root / "resnet50_maskrcnn" / "template.yaml")
default_templates = [default_template]
default_templates_ids = [default_template.model_template_id]

templates = Registry("src/otx/algorithms/detection").filter(task_type="INSTANCE_SEGMENTATION").templates
templates_ids = [template.model_template_id for template in templates]

# Add experimental templates
templates_with_experimental = copy.deepcopy(templates)
templates_ids_with_experimental = copy.deepcopy(templates_ids)
for experimental_template in iseg_config_root.glob("**/*_experimental.yaml"):
    template_experimental = parse_model_template(experimental_template)
    templates_with_experimental.extend([template_experimental])
    templates_ids_with_experimental.extend([template_experimental.model_template_id])


TestInstanceSegmentationModelTemplates = generate_model_template_testing(templates)


class TestInstanceSegmentationCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_train_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_resume(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg/test_resume"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args)
        args1["train_params"] = resume_params
        args1[
            "--resume-from"
        ] = f"{template_work_dir}/trained_for_resume_{template.model_template_id}/models/weights.pth"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    @pytest.mark.parametrize("dump_features", [True, False])
    def test_otx_export(self, template, tmp_dir_path, dump_features):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_export_testing(template, tmp_dir_path, dump_features)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_export_fp16(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_export_testing(template, tmp_dir_path, half_precision=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_export_onnx(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_export_testing(template, tmp_dir_path, half_precision=False, is_onnx=True, check_ir_meta=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    @pytest.mark.parametrize("half_precision", [True, False])
    def test_otx_eval_openvino(self, template, tmp_dir_path, half_precision):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=1.0, half_precision=half_precision)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_explain(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_explain_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_explain_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_explain_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args, threshold=1.0)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_hpo(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg/test_hpo"
        otx_hpo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")
        if torch.__version__.startswith("2.") and template.name.startswith("MaskRCNN"):
            pytest.skip(
                reason="Issue#2451: Torch2.0 CUDA runtime error during NNCF optimization of ROIAlign MMCV kernel for MaskRCNN"
            )
        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_multi_gpu_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg/test_multi_gpu"
        args1 = copy.deepcopy(args)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_train_semisl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg/test_semisl"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl)
        template_dir = get_template_dir(template, tmp_dir_path)
        assert (Path(template_dir) / "semisl").is_dir()

    @e2e_pytest_component
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_multi_gpu_train_semisl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg/test_multi_gpu_semisl"
        args_semisl_multigpu = copy.deepcopy(args_semisl)
        args_semisl_multigpu["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl_multigpu)
        template_dir = get_template_dir(template, tmp_dir_path)
        assert (Path(template_dir) / "semisl").is_dir()
