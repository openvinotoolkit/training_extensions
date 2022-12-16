# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import os.path as osp

import torch
import pytest

from mpa_tasks.apis.classification import ClassificationInferenceTask, ClassificationTrainTask
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_api
from ote_cli.utils.io import save_model_data, read_model
from tests.api_tests.test_ote_classification_api import MPAClsAPIBase, DEFAULT_CLS_TEMPLATE_DIR
from tests.api_tests.test_ote_detection_api import MPADetAPIBase, DEFAULT_DET_TEMPLATE_DIR
from torchreid_tasks.openvino_task import OpenVINOClassificationTask
from detection_tasks.apis.detection.openvino_task import OpenVINODetectionTask
from mpa_tasks.apis.detection import DetectionInferenceTask, DetectionTrainTask


torch.manual_seed(0)

save_model_to = "/tmp/ote_xai/"

assert_text_torch = "For the torch task the number of saliency maps should be equal to the number of all classes."
assert_text_ov = "For the OV task the number of saliency maps should be equal to the number of predicted classes."


def saliency_maps_check(predicted_dataset, task_labels, assert_text, only_predicted=False):
    for data_point in predicted_dataset:
        saliency_map_counter = 0
        metadata_list = data_point.get_metadata()
        for metadata in metadata_list:
            if isinstance(metadata.data, ResultMediaEntity):
                if metadata.data.type == "saliency_map":
                    saliency_map_counter += 1
                    assert metadata.data.numpy.ndim == 3
                    assert metadata.data.numpy.shape == (data_point.height, data_point.width, 3)
        if only_predicted:
            assert saliency_map_counter == len(data_point.get_roi_labels(task_labels)), assert_text
        else:
            assert saliency_map_counter == len(task_labels), assert_text


class TestOVClsXAIAPI(MPAClsAPIBase):
    @e2e_pytest_api
    @pytest.mark.parametrize(
        "multilabel,hierarchical",
        [(False, False), (True, False), (False, True)],
        ids=["multiclass", "multilabel", "hierarchical"],
    )
    def test_inference_xai(self, multilabel, hierarchical):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_CLS_TEMPLATE_DIR, num_iters=1)
        task_environment, dataset = self.init_environment(
            hyper_parameters, model_template, multilabel, hierarchical, 20
        )

        # Train and save a model
        task = ClassificationTrainTask(task_environment=task_environment)
        train_parameters = TrainParameters
        output_model = ModelEntity(
            dataset,
            task_environment.get_model_configuration(),
        )
        task.train(dataset, output_model, train_parameters)
        save_model_data(output_model, save_model_to)

        # Infer torch model
        task = ClassificationInferenceTask(task_environment=task_environment)
        predicted_dataset = task.infer(dataset.with_empty_annotations(), InferenceParameters)

        # Check saliency maps torch task
        task_labels = output_model.configuration.get_label_schema().get_labels(include_empty=False)
        saliency_maps_check(predicted_dataset, task_labels, assert_text_torch)

        # Save OV IR model
        task._model_ckpt = osp.join(save_model_to, "weights.pth")
        exported_model = ModelEntity(None, task_environment.get_model_configuration())
        task.export(ExportType.OPENVINO, exported_model)
        os.makedirs(save_model_to, exist_ok=True)
        save_model_data(exported_model, save_model_to)

        # Infer OV IR model
        load_weights_ov = osp.join(save_model_to, "openvino.xml")
        task_environment.model = read_model(task_environment.get_model_configuration(), load_weights_ov, None)
        task = OpenVINOClassificationTask(task_environment=task_environment)
        predicted_dataset_ov = task.infer(
            dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=False),
        )

        # Check saliency maps OV task
        saliency_maps_check(predicted_dataset_ov, task_labels, assert_text_ov, only_predicted=True)


class TestOVDetXAIAPI(MPADetAPIBase):
    @e2e_pytest_api
    def test_inference_xai(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_DET_TEMPLATE_DIR, num_iters=2)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 10)

        train_task = DetectionTrainTask(task_environment=detection_environment)
        trained_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
        )
        train_task.train(dataset, trained_model, TrainParameters)
        save_model_data(trained_model, save_model_to)

        # Infer torch model
        detection_environment.model = trained_model
        inference_task = DetectionInferenceTask(task_environment=detection_environment)
        predicted_dataset = inference_task.infer(dataset.with_empty_annotations())

        # Check saliency maps torch task
        task_labels = trained_model.configuration.get_label_schema().get_labels(include_empty=False)
        saliency_maps_check(predicted_dataset, task_labels, assert_text_torch)

        # Save OV IR model
        inference_task._model_ckpt = osp.join(save_model_to, "weights.pth")
        exported_model = ModelEntity(None, detection_environment.get_model_configuration())
        inference_task.export(ExportType.OPENVINO, exported_model)
        os.makedirs(save_model_to, exist_ok=True)
        save_model_data(exported_model, save_model_to)

        # Infer OV IR model
        load_weights_ov = osp.join(save_model_to, "openvino.xml")
        detection_environment.model = read_model(detection_environment.get_model_configuration(), load_weights_ov, None)
        task = OpenVINODetectionTask(task_environment=detection_environment)
        predicted_dataset_ov = task.infer(
            dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=False),
        )

        # Check saliency maps OV task
        saliency_maps_check(predicted_dataset_ov, task_labels, assert_text_ov, only_predicted=True)
