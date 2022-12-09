# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import os.path as osp

import numpy as np
import torch
import pytest

from mmcls.models import build_classifier
from mmdet.models import build_detector

from mpa.det.stage import DetectionStage  # noqa
from mpa.modules.hooks.auxiliary_hooks import ReciproCAMHook, DetSaliencyMapHook
from mpa.utils.config_utils import MPAConfig
from mpa_tasks.apis.classification import ClassificationInferenceTask, ClassificationTrainTask
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_api
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component
from ote_cli.utils.io import save_model_data, read_model
from ote_cli.registry import Registry
from tests.api_tests.test_ote_classification_api import TestMPAClsAPI, DEFAULT_CLS_TEMPLATE_DIR
from tests.api_tests.test_ote_detection_api import TestMPADetAPI, DEFAULT_DET_TEMPLATE_DIR
from torchreid_tasks.openvino_task import OpenVINOClassificationTask
from detection_tasks.apis.detection.openvino_task import OpenVINODetectionTask
from mpa_tasks.apis.detection import DetectionInferenceTask, DetectionTrainTask


torch.manual_seed(0)

save_model_to = "/tmp/ote_xai/"

assert_text_torch = "For the torch task the number of saliency maps should be equal to the number of all classes."
assert_text_ov = "For the OV task the number of saliency maps should be equal to the number of predicted classes."

templates_cls = Registry("external/model-preparation-algorithm").filter(task_type="CLASSIFICATION").templates
templates_cls_ids = [template.model_template_id for template in templates_cls]

templates_det = Registry("external/model-preparation-algorithm").filter(task_type="DETECTION").templates
templates_det_ids = [template.model_template_id for template in templates_det]


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


class TestExplainMethods:
    ref_saliency_vals_cls = {
        "EfficientNet-B0": np.array([36, 185, 190, 159, 173, 124, 19], dtype=np.uint8),
        "MobileNet-V3-large-1x": np.array([21, 38, 56, 134, 100, 41, 38], dtype=np.uint8),
        "EfficientNet-V2-S": np.array([166, 204, 201, 206, 218, 221, 138], dtype=np.uint8),
    }

    ref_saliency_shapes = {
        "ATSS": (2, 4, 4),
        "SSD": (81, 13, 13),
        "YOLOX": (80, 13, 13),
    }

    ref_saliency_vals_det = {
        "ATSS": np.array([78, 217, 42, 102], dtype=np.uint8),
        "SSD": np.array([225, 158, 221, 106, 146, 158, 227, 149, 137, 135, 200, 159, 255], dtype=np.uint8),
        "YOLOX": np.array([109, 174, 82, 214, 178, 184, 168, 161, 163, 156, 220, 233, 195], dtype=np.uint8),
    }

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_cls, ids=templates_cls_ids)
    def test_saliency_map_cls(self, template):
        base_dir = os.path.abspath(os.path.dirname(template.model_template_path))
        cfg_path = os.path.join(base_dir, "model.py")
        cfg = MPAConfig.fromfile(cfg_path)

        cfg.model.pop("task")
        model = build_classifier(cfg.model)
        model = model.eval()

        img = torch.rand(2, 3, 224, 224) - 0.5
        data = {"img_metas": {}, "img": img}

        with ReciproCAMHook(model) as rcam_hook:
            with torch.no_grad():
                _ = model(return_loss=False, **data)
        saliency_maps = rcam_hook.records

        assert len(saliency_maps) == 2
        assert saliency_maps[0].ndim == 3
        assert saliency_maps[0].shape == (1000, 7, 7)

        assert (saliency_maps[0][0][0] == self.ref_saliency_vals_cls[template.name]).all()

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_det, ids=templates_det_ids)
    def test_saliency_map_det(self, template):
        base_dir = os.path.abspath(os.path.dirname(template.model_template_path))
        cfg_path = os.path.join(base_dir, "model.py")
        cfg = MPAConfig.fromfile(cfg_path)

        model = build_detector(cfg.model)
        model = model.eval()

        img = torch.rand(2, 3, 416, 416) - 0.5
        img_metas = [
            {
                "img_shape": (416, 416, 3),
                "scale_factor": np.array([1.1784703, 0.832, 1.1784703, 0.832], dtype=np.float32),
            },
            {
                "img_shape": (416, 416, 3),
                "scale_factor": np.array([1.1784703, 0.832, 1.1784703, 0.832], dtype=np.float32),
            },
        ]
        data = {"img_metas": [img_metas], "img": [img]}

        with DetSaliencyMapHook(model) as det_hook:
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **data)
        saliency_maps = det_hook.records

        assert len(saliency_maps) == 2
        assert saliency_maps[0].ndim == 3
        assert saliency_maps[0].shape == self.ref_saliency_shapes[template.name]
        assert (saliency_maps[0][0][0] == self.ref_saliency_vals_det[template.name]).all()


class TestOVClsXAIAPI(TestMPAClsAPI):
    @e2e_pytest_api
    @pytest.mark.parametrize(
        "multilabel,hierarchical",
        [(False, False), (True, False), (False, True)],
        ids=["multiclass", "multilabel", "hierarchical"],
    )
    def test_inference_xai(self, multilabel, hierarchical):
        hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_CLS_TEMPLATE_DIR, num_iters=1)
        multilabel = False
        hierarchical = True
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


class TestOVDetXAIAPI(TestMPADetAPI):
    @e2e_pytest_api
    def test_inference_xai(self):
        save_model_to = "/tmp/ote_xai/"
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
