# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import os.path as osp
import tempfile
from copy import deepcopy

import pytest
import torch

from otx.algorithms.classification.adapters.mmcls.task import MMClassificationTask
from otx.algorithms.classification.adapters.openvino.task import ClassificationOpenVINOTask

from otx.algorithms.detection.adapters.mmdet.task import MMDetectionTask
from otx.algorithms.detection.adapters.openvino.task import OpenVINODetectionTask
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.api.configuration.helper import create
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import (
    ModelConfiguration,
    ModelEntity,
)
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.subset import Subset
from otx.api.entities.train_parameters import TrainParameters
from otx.api.entities.model_template import parse_model_template, TaskType
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.cli.utils.io import read_model, save_model_data
from tests.integration.api.classification.test_api_classification import (
    DEFAULT_CLS_TEMPLATE_DIR,
    ClassificationTaskAPIBase,
)
from tests.integration.api.detection.api_detection import DetectionTaskAPIBase, DEFAULT_DET_TEMPLATE_DIR
from tests.test_suite.e2e_test_system import e2e_pytest_api
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_ISEG_TEMPLATE_DIR,
    init_environment,
    generate_det_dataset,
)

torch.manual_seed(0)

assert_text_explain_all = "The number of saliency maps should be equal to the number of all classes."
assert_text_explain_predicted = "The number of saliency maps should be equal to the number of predicted classes."


def saliency_maps_check(
    predicted_dataset, task_labels, raw_sal_map_shape, processed_saliency_maps=False, only_predicted=True
):
    for data_point in predicted_dataset:
        saliency_map_counter = 0
        metadata_list = data_point.get_metadata()
        for metadata in metadata_list:
            if isinstance(metadata.data, ResultMediaEntity):
                if metadata.data.type == "saliency_map":
                    saliency_map_counter += 1
                    if processed_saliency_maps:
                        assert metadata.data.numpy.ndim == 3
                        assert metadata.data.numpy.shape == (data_point.height, data_point.width, 3)
                    else:
                        assert metadata.data.numpy.ndim == 2
                        assert metadata.data.numpy.shape == raw_sal_map_shape
        if only_predicted:
            assert saliency_map_counter == len(data_point.annotation_scene.get_labels()), assert_text_explain_predicted
        else:
            assert saliency_map_counter == len(task_labels), assert_text_explain_all


class TestOVClsXAIAPI(ClassificationTaskAPIBase):
    ref_raw_saliency_shapes = {
        "EfficientNet-B0": (7, 7),
    }

    @e2e_pytest_api
    @pytest.mark.parametrize(
        "multilabel,hierarchical",
        [(False, False), (True, False), (False, True)],
        ids=["multiclass", "multilabel", "hierarchical"],
    )
    def test_inference_xai(self, multilabel, hierarchical):
        with tempfile.TemporaryDirectory() as temp_dir:
            hyper_parameters, model_template = self.setup_configurable_parameters(DEFAULT_CLS_TEMPLATE_DIR, num_iters=1)
            task_environment, dataset = self.init_environment(
                hyper_parameters, model_template, multilabel, hierarchical, 20
            )

            # Train and save a model
            task = MMClassificationTask(task_environment=task_environment)
            train_parameters = TrainParameters()
            output_model = ModelEntity(
                dataset,
                task_environment.get_model_configuration(),
            )
            task.train(dataset, output_model, train_parameters)
            save_model_data(output_model, temp_dir)

            for processed_saliency_maps, only_predicted in [[True, False], [False, True]]:
                task_environment, dataset = self.init_environment(
                    hyper_parameters, model_template, multilabel, hierarchical, 20
                )

                # Infer torch model
                task = MMClassificationTask(task_environment=task_environment)
                inference_parameters = InferenceParameters(
                    is_evaluation=False,
                    process_saliency_maps=processed_saliency_maps,
                    explain_predicted_classes=only_predicted,
                )
                predicted_dataset = task.infer(dataset.with_empty_annotations(), inference_parameters)

                # Check saliency maps torch task
                task_labels = output_model.configuration.get_label_schema().get_labels(include_empty=False)
                saliency_maps_check(
                    predicted_dataset,
                    task_labels,
                    self.ref_raw_saliency_shapes[model_template.name],
                    processed_saliency_maps=processed_saliency_maps,
                    only_predicted=only_predicted,
                )

                # Save OV IR model
                task._model_ckpt = osp.join(temp_dir, "weights.pth")
                exported_model = ModelEntity(None, task_environment.get_model_configuration())
                task.export(ExportType.OPENVINO, exported_model, dump_features=True)
                os.makedirs(temp_dir, exist_ok=True)
                save_model_data(exported_model, temp_dir)

                # Infer OV IR model
                load_weights_ov = osp.join(temp_dir, "openvino.xml")
                task_environment.model = read_model(task_environment.get_model_configuration(), load_weights_ov, None)
                task = ClassificationOpenVINOTask(task_environment=task_environment)
                _, dataset = self.init_environment(hyper_parameters, model_template, multilabel, hierarchical, 20)
                predicted_dataset_ov = task.infer(dataset.with_empty_annotations(), inference_parameters)

                # Check saliency maps OV task
                saliency_maps_check(
                    predicted_dataset_ov,
                    task_labels,
                    self.ref_raw_saliency_shapes[model_template.name],
                    processed_saliency_maps=processed_saliency_maps,
                    only_predicted=only_predicted,
                )


class TestOVDetXAIAPI(DetectionTaskAPIBase):
    ref_raw_saliency_shapes = {
        "MobileNetV2-ATSS": (6, 8),
    }

    @e2e_pytest_api
    def test_inference_xai(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            hyper_parameters, model_template = self.setup_configurable_parameters(
                DEFAULT_DET_TEMPLATE_DIR, num_iters=15
            )
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


class TestOVDetTilXAIAPI(DetectionTaskAPIBase):
    ref_raw_saliency_shapes = {
        "MobileNetV2-ATSS": (6, 8),
    }

    @e2e_pytest_api
    def test_inference_xai(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            hyper_parameters, model_template = self.setup_configurable_parameters(
                DEFAULT_DET_TEMPLATE_DIR, num_iters=10, tiling=True
            )
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


class TestOVISegmXAIAPI:
    @e2e_pytest_api
    def test_inference_xai(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
            hyper_parameters = create(model_template.hyper_parameters.data)
            hyper_parameters.learning_parameters.num_iters = 3
            task_env = init_environment(hyper_parameters, model_template, task_type=TaskType.INSTANCE_SEGMENTATION)

            train_task = MMDetectionTask(task_env)

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

            train_task.train(iseg_dataset, trained_model, TrainParameters())

            save_model_data(trained_model, temp_dir)

            processed_saliency_maps, only_predicted = False, True
            task_env = init_environment(hyper_parameters, model_template, task_type=TaskType.INSTANCE_SEGMENTATION)
            inference_parameters = InferenceParameters(
                is_evaluation=False,
                process_saliency_maps=processed_saliency_maps,
                explain_predicted_classes=only_predicted,
            )

            # Infer torch model
            task_env.model = trained_model
            inference_task = MMDetectionTask(task_environment=task_env)
            val_dataset = iseg_dataset.get_subset(Subset.VALIDATION)
            val_dataset_copy = deepcopy(val_dataset)
            predicted_dataset = inference_task.infer(val_dataset.with_empty_annotations(), inference_parameters)

            # Check saliency maps torch task
            task_labels = trained_model.configuration.get_label_schema().get_labels(include_empty=False)
            saliency_maps_check(
                predicted_dataset,
                task_labels,
                (224, 224),
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
            predicted_dataset_ov = task.infer(val_dataset_copy.with_empty_annotations(), inference_parameters)

            # Check saliency maps OV task
            saliency_maps_check(
                predicted_dataset_ov,
                task_labels,
                (480, 640),
                processed_saliency_maps=processed_saliency_maps,
                only_predicted=only_predicted,
            )


class TestOVISegmTilXAIAPI:
    @e2e_pytest_api
    def test_inference_xai(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
            hyper_parameters = create(model_template.hyper_parameters.data)
            hyper_parameters.learning_parameters.num_iters = 5
            hyper_parameters.tiling_parameters.enable_tiling = True
            task_env = init_environment(hyper_parameters, model_template, task_type=TaskType.INSTANCE_SEGMENTATION)

            train_task = MMDetectionTask(task_env)

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

            train_task.train(iseg_dataset, trained_model, TrainParameters())

            save_model_data(trained_model, temp_dir)

            processed_saliency_maps, only_predicted = False, True
            task_env = init_environment(hyper_parameters, model_template, task_type=TaskType.INSTANCE_SEGMENTATION)
            inference_parameters = InferenceParameters(
                is_evaluation=False,
                process_saliency_maps=processed_saliency_maps,
                explain_predicted_classes=only_predicted,
            )

            # Infer torch model
            task_env.model = trained_model
            inference_task = MMDetectionTask(task_environment=task_env)
            val_dataset = iseg_dataset.get_subset(Subset.VALIDATION)
            val_dataset_copy = deepcopy(val_dataset)
            predicted_dataset = inference_task.infer(val_dataset.with_empty_annotations(), inference_parameters)

            # Check saliency maps torch task
            task_labels = trained_model.configuration.get_label_schema().get_labels(include_empty=False)
            saliency_maps_check(
                predicted_dataset,
                task_labels,
                (33, 44),
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
            predicted_dataset_ov = task.infer(val_dataset_copy.with_empty_annotations(), inference_parameters)

            # Check saliency maps OV task
            saliency_maps_check(
                predicted_dataset_ov,
                task_labels,
                (480, 640),
                processed_saliency_maps=processed_saliency_maps,
                only_predicted=only_predicted,
            )
