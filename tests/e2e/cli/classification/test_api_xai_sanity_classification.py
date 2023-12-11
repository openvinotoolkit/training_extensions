# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import os.path as osp
import tempfile

import pytest
import torch
import numpy as np

from otx.algorithms.classification.adapters.mmcls.task import MMClassificationTask
from otx.algorithms.classification.adapters.openvino.task import ClassificationOpenVINOTask

from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import (
    ModelEntity,
)
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.cli.utils.io import read_model, save_model_data
from tests.integration.api.classification.test_api_classification import (
    DEFAULT_CLS_TEMPLATE_DIR,
    ClassificationTaskAPIBase,
)
from tests.test_suite.e2e_test_system import e2e_pytest_api

torch.manual_seed(0)

assert_text_explain_all = "The number of saliency maps should be equal to the number of all classes."
assert_text_explain_predicted = "The number of saliency maps should be equal to the number of predicted classes."


def saliency_maps_check(
    predicted_dataset, task_labels, raw_sal_map_shape=None, processed_saliency_maps=False, only_predicted=True
):
    for data_point in predicted_dataset:
        saliency_map_counter = 0
        metadata_list = data_point.get_metadata()
        for metadata in metadata_list:
            if isinstance(metadata.data, ResultMediaEntity):
                if metadata.data.type == "saliency_map":
                    saliency_map_counter += 1
                    if processed_saliency_maps:
                        assert metadata.data.numpy.ndim == 3, "Number of dims is incorrect."
                        assert metadata.data.numpy.shape == (data_point.height, data_point.width, 3)
                    else:
                        assert metadata.data.numpy.ndim == 2, "Raw saliency map has to be two-dimensional."
                        if raw_sal_map_shape:
                            assert (
                                metadata.data.numpy.shape == raw_sal_map_shape
                            ), "Raw saliency map shape is incorrect."
                    assert metadata.data.numpy.dtype == np.uint8, "Saliency map has to be uint8 dtype."
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
