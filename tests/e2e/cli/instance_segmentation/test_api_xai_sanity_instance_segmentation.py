# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import os.path as osp
import pytest

import torch

from otx.algorithms.common.utils.utils import is_xpu_available
from otx.algorithms.detection.adapters.mmdet.task import MMDetectionTask
from otx.algorithms.detection.adapters.openvino.task import OpenVINODetectionTask
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.api.configuration.helper import create
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import (
    ModelConfiguration,
    ModelEntity,
)
from otx.api.entities.subset import Subset
from otx.api.entities.train_parameters import TrainParameters
from otx.api.entities.model_template import parse_model_template, TaskType
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.cli.utils.io import read_model, save_model_data
from tests.e2e.cli.classification.test_api_xai_sanity_classification import saliency_maps_check
from tests.test_suite.e2e_test_system import e2e_pytest_api
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_ISEG_TEMPLATE_DIR,
    init_environment,
    generate_det_dataset,
)

torch.manual_seed(0)

assert_text_explain_all = "The number of saliency maps should be equal to the number of all classes."
assert_text_explain_predicted = "The number of saliency maps should be equal to the number of predicted classes."

if is_xpu_available():
    pytest.skip("Instance segmentation task is not supported on XPU", allow_module_level=True)


class TestISegmXAIAPI:
    def _prepare_task_env(self, temp_dir, train=True, tile=False):
        model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.num_iters = 5
        if tile:
            hyper_parameters.tiling_parameters.enable_tiling = True
        task_env = init_environment(hyper_parameters, model_template, task_type=TaskType.INSTANCE_SEGMENTATION)

        iseg_dataset, iseg_labels = generate_det_dataset(TaskType.INSTANCE_SEGMENTATION, 100)
        iseg_label_schema = LabelSchemaEntity()
        iseg_label_group = LabelGroup(
            name="labels",
            labels=iseg_labels,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        iseg_label_schema.add_group(iseg_label_group)

        _config = ModelConfiguration(DetectionConfig(), iseg_label_schema)
        trained_model = ModelEntity(
            iseg_dataset,
            _config,
        )

        if train:
            train_task = MMDetectionTask(task_env)
            train_task.train(iseg_dataset, trained_model, TrainParameters())

        save_model_data(trained_model, temp_dir)
        task_env = init_environment(hyper_parameters, model_template, task_type=TaskType.INSTANCE_SEGMENTATION)
        task_env.model = trained_model

        return task_env, iseg_dataset

    def _export_and_save_ov_ir_model(self, temp_dir, task_env, inference_task):
        inference_task._model_ckpt = osp.join(temp_dir, "weights.pth")
        exported_model = ModelEntity(None, task_env.get_model_configuration())
        inference_task.export(ExportType.OPENVINO, exported_model, dump_features=True)
        os.makedirs(temp_dir, exist_ok=True)
        save_model_data(exported_model, temp_dir)

    @e2e_pytest_api
    @pytest.mark.parametrize("tile", [True, False])
    def test_torch_xai_inference(self, tile, tmp_dir_path):
        if tile:
            tmp_dir_path = tmp_dir_path / "tile"
        else:
            tmp_dir_path = tmp_dir_path / "no_tile"
        task_env, iseg_dataset = self._prepare_task_env(tmp_dir_path, tile=tile)
        inference_task = MMDetectionTask(task_environment=task_env)
        val_dataset = iseg_dataset.get_subset(Subset.VALIDATION)
        inference_parameters = InferenceParameters(
            is_evaluation=False,
        )
        predicted_dataset = inference_task.infer(val_dataset.with_empty_annotations(), inference_parameters)
        task_labels = task_env.model.configuration.get_label_schema().get_labels(include_empty=False)

        if tile:
            ref_shape = None
        else:
            ref_shape = (28, 28)
        saliency_maps_check(
            predicted_dataset,
            task_labels,
            ref_shape,
            processed_saliency_maps=False,
            only_predicted=True,
        )
        self._export_and_save_ov_ir_model(tmp_dir_path, task_env, inference_task)

    @pytest.mark.parametrize(
        "tile, enable_async_inference", [[True, True], [True, False], [False, False], [False, True]]
    )
    def test_ov_xai_inference(self, tile, enable_async_inference, tmp_dir_path):
        if tile:
            tmp_dir_path = tmp_dir_path / "tile"
            pytest.skip(reason="[Issue#2434] Need to fix merging sailency map")
        else:
            tmp_dir_path = tmp_dir_path / "no_tile"
        task_env, iseg_dataset = self._prepare_task_env(tmp_dir_path, train=False, tile=tile)
        load_weights_ov = osp.join(tmp_dir_path, "openvino.xml")
        task_env.model = read_model(task_env.get_model_configuration(), load_weights_ov, None)
        task = OpenVINODetectionTask(task_environment=task_env)
        inference_parameters = InferenceParameters(
            is_evaluation=False,
        )
        inference_parameters.enable_async_inference = enable_async_inference
        val_dataset = iseg_dataset.get_subset(Subset.VALIDATION)
        predicted_dataset_ov = task.infer(val_dataset.with_empty_annotations(), inference_parameters)
        task_labels = task_env.model.configuration.get_label_schema().get_labels(include_empty=False)

        saliency_maps_check(
            predicted_dataset_ov,
            task_labels,
            (480, 640),
            processed_saliency_maps=False,
            only_predicted=True,
        )
