# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import os.path as osp
import tempfile

import torch

from otx.algorithms.common.configs.configuration_enums import InputSizePreset
from otx.algorithms.common.utils import set_random_seed
from otx.algorithms.detection.adapters.mmdet.task import MMDetectionTask
from otx.algorithms.detection.adapters.openvino.task import OpenVINODetectionTask
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import (
    ModelEntity,
)
from otx.api.entities.subset import Subset
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.cli.utils.io import read_model, save_model_data
from tests.e2e.cli.classification.test_api_xai_sanity_classification import saliency_maps_check
from tests.integration.api.detection.api_detection import DetectionTaskAPIBase, DEFAULT_DET_TEMPLATE_DIR
from tests.test_suite.e2e_test_system import e2e_pytest_api

set_random_seed(0)

assert_text_explain_all = "The number of saliency maps should be equal to the number of all classes."
assert_text_explain_predicted = "The number of saliency maps should be equal to the number of predicted classes."


class TestOVDetXAIAPI(DetectionTaskAPIBase):
    ref_raw_saliency_shapes = {
        "MobileNetV2-ATSS": (16, 16),  # Need to be adapted to configurable or adaptive input size
    }

    @e2e_pytest_api
    def test_inference_xai(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            hyper_parameters, model_template = self.setup_configurable_parameters(
                DEFAULT_DET_TEMPLATE_DIR, num_iters=15
            )
            hyper_parameters.learning_parameters.input_size = InputSizePreset._512x512  # To fix saliency map size
            task_env, dataset = self.init_environment(hyper_parameters, model_template, 10)

            train_task = MMDetectionTask(task_environment=task_env)
            trained_model = ModelEntity(
                dataset,
                task_env.get_model_configuration(),
            )
            train_task.train(dataset, trained_model, TrainParameters())
            save_model_data(trained_model, temp_dir)

            for processed_saliency_maps, only_predicted in [[True, False], [False, True]]:
                task_env, dataset = self.init_environment(hyper_parameters, model_template, 10)
                inference_parameters = InferenceParameters(
                    is_evaluation=False,
                    process_saliency_maps=processed_saliency_maps,
                    explain_predicted_classes=only_predicted,
                )

                # Infer torch model
                task_env.model = trained_model
                inference_task = MMDetectionTask(task_environment=task_env)
                val_dataset = dataset.get_subset(Subset.VALIDATION)
                predicted_dataset = inference_task.infer(val_dataset.with_empty_annotations(), inference_parameters)

                # Check saliency maps torch task
                task_labels = trained_model.configuration.get_label_schema().get_labels(include_empty=False)
                saliency_maps_check(
                    predicted_dataset,
                    task_labels,
                    self.ref_raw_saliency_shapes[model_template.name],
                    processed_saliency_maps=processed_saliency_maps,
                    only_predicted=only_predicted,
                )

                # Save OV IR model
                inference_task._model_ckpt = osp.join(temp_dir, "weights.pth")
                exported_model = ModelEntity(None, task_env.get_model_configuration())
                inference_task.export(ExportType.OPENVINO, exported_model, dump_features=True)
                os.makedirs(temp_dir, exist_ok=True)
                save_model_data(exported_model, temp_dir)

                # Infer OV IR model
                load_weights_ov = osp.join(temp_dir, "openvino.xml")
                task_env.model = read_model(task_env.get_model_configuration(), load_weights_ov, None)
                task = OpenVINODetectionTask(task_environment=task_env)
                _, dataset = self.init_environment(hyper_parameters, model_template, 10)
                predicted_dataset_ov = task.infer(dataset.with_empty_annotations(), inference_parameters)

                # Check saliency maps OV task
                saliency_maps_check(
                    predicted_dataset_ov,
                    task_labels,
                    self.ref_raw_saliency_shapes[model_template.name],
                    processed_saliency_maps=processed_saliency_maps,
                    only_predicted=only_predicted,
                )


# disable test until fix in PR#2337 is merged
# class TestOVDetTilXAIAPI(DetectionTaskAPIBase):
#     ref_raw_saliency_shapes = {
#         "ATSS": (6, 8),
#     }

#     @e2e_pytest_api
#     def test_inference_xai(self):
#         with tempfile.TemporaryDirectory() as temp_dir:
#             hyper_parameters, model_template = self.setup_configurable_parameters(
#                 DEFAULT_DET_TEMPLATE_DIR, num_iters=10, tiling=True
#             )
#             task_env, dataset = self.init_environment(hyper_parameters, model_template, 10)

#             train_task = MMDetectionTask(task_environment=task_env)
#             trained_model = ModelEntity(
#                 dataset,
#                 task_env.get_model_configuration(),
#             )
#             train_task.train(dataset, trained_model, TrainParameters())
#             save_model_data(trained_model, temp_dir)

#             for processed_saliency_maps, only_predicted in [[True, False], [False, True]]:
#                 task_env, dataset = self.init_environment(hyper_parameters, model_template, 10)
#                 inference_parameters = InferenceParameters(
#                     is_evaluation=False,
#                     process_saliency_maps=processed_saliency_maps,
#                     explain_predicted_classes=only_predicted,
#                 )

#                 # Infer torch model
#                 task_env.model = trained_model
#                 inference_task = MMDetectionTask(task_environment=task_env)
#                 val_dataset = dataset.get_subset(Subset.VALIDATION)
#                 predicted_dataset = inference_task.infer(val_dataset.with_empty_annotations(), inference_parameters)

#                 # Check saliency maps torch task
#                 task_labels = trained_model.configuration.get_label_schema().get_labels(include_empty=False)
#                 saliency_maps_check(
#                     predicted_dataset,
#                     task_labels,
#                     self.ref_raw_saliency_shapes[model_template.name],
#                     processed_saliency_maps=processed_saliency_maps,
#                     only_predicted=only_predicted,
#                 )

#                 # Save OV IR model
#                 inference_task._model_ckpt = osp.join(temp_dir, "weights.pth")
#                 exported_model = ModelEntity(None, task_env.get_model_configuration())
#                 inference_task.export(ExportType.OPENVINO, exported_model, dump_features=True)
#                 os.makedirs(temp_dir, exist_ok=True)
#                 save_model_data(exported_model, temp_dir)

#                 # Infer OV IR model
#                 load_weights_ov = osp.join(temp_dir, "openvino.xml")
#                 task_env.model = read_model(task_env.get_model_configuration(), load_weights_ov, None)
#                 task = OpenVINODetectionTask(task_environment=task_env)
#                 _, dataset = self.init_environment(hyper_parameters, model_template, 10)
#                 inference_parameters.enable_async_inference = False
#                 predicted_dataset_ov = task.infer(dataset.with_empty_annotations(), inference_parameters)

#                 # Check saliency maps OV task
#                 saliency_maps_check(
#                     predicted_dataset_ov,
#                     task_labels,
#                     self.ref_raw_saliency_shapes[model_template.name],
#                     processed_saliency_maps=processed_saliency_maps,
#                     only_predicted=only_predicted,
#                 )
