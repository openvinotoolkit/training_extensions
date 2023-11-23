"""Tests for OTX Class-Incremental Learning for instance segmentation with OTX CLI"""
# Copyright (C) 2022-2023-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import os

import pytest
import torch

from otx.api.entities.model_template import parse_model_template
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
    otx_train_testing,
    otx_resume_testing,
)

args = {
    "--train-data-roots": "tests/assets/small_objects",
    "--val-data-roots": "tests/assets/small_objects",
    "--test-data-roots": "tests/assets/small_objects",
    "--input": "tests/assets/small_objects/images/train",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "1",
        "--learning_parameters.batch_size",
        "4",
        "--tiling_parameters.enable_tiling",
        "1",
        "--tiling_parameters.enable_tile_classifier",
        "1",
        "--tiling_parameters.enable_adaptive_params",
        "1",
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
    "4",
    "--tiling_parameters.enable_tiling",
    "1",
    "--tiling_parameters.enable_tile_classifier",
    "1",
    "--tiling_parameters.enable_adaptive_params",
    "1",
]

otx_dir = os.getcwd()

default_template = parse_model_template(
    os.path.join("src/otx/algorithms/detection/configs", "instance_segmentation", "resnet50_maskrcnn", "template.yaml")
)
templates = [default_template]
templates_ids = [default_template.model_template_id]


class TestTilingInstanceSegmentationCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg"
        otx_train_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_resume(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg/test_resume"
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
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg"
        otx_export_testing(template, tmp_dir_path, dump_features)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export_fp16(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg"
        otx_export_testing(template, tmp_dir_path, half_precision=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export_onnx(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg"
        otx_export_testing(template, tmp_dir_path, half_precision=False, is_onnx=True, check_ir_meta=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("half_precision", [True, False])
    def test_otx_eval_openvino(self, template, tmp_dir_path, half_precision):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=1.0, half_precision=half_precision)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg"
        otx_explain_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg"
        otx_explain_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args, threshold=1.0)

    @e2e_pytest_component
    @pytest.mark.skip("Tiling w/ HPO fails in CI")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_hpo(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg/test_hpo"
        otx_hpo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "tiling_ins_seg"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")
        if torch.__version__.startswith("2.") and template.name.startswith("MaskRCNN"):
            pytest.skip(
                reason="Issue#2451: Torch2.0 CUDA runtime error during NNCF optimization of ROIAlign MMCV kernel for MaskRCNN"
            )
        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)
