"""Test otx segmentation task."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import numpy as np
import pytest

from otx.algorithms.segmentation.task import OTXSegmentationTask
from otx.api.configuration.helper import create
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model_template import (
    parse_model_template,
)
from tests.unit.algorithms.segmentation.test_helpers import (
    DEFAULT_SEG_TEMPLATE_DIR,
    generate_otx_dataset,
    generate_otx_label_schema,
    init_environment,
)
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockOTXSegmentationTask(OTXSegmentationTask):
    def _infer_model(*args, **kwargs):
        return dict(
            classes=["background", "rectangle", "ellipse", "triangle"],
            eval_predictions=[[np.random.rand(4, 128, 128)]],
            feature_vectors=[np.random.rand(600, 1, 1)],
        )

    def _train_model(*args, **kwargs):
        return {"final_ckpt": "dummy.pth"}

    def _explain_model(*args, **kwargs):
        pass

    def _export_model(*args, **kwargs):
        return {
            "outputs": {"bin": f"/tmp/model.xml", "xml": f"/tmp/model.bin", "onnx": f"/tmp/model.onnx"},
            "inference_parameters": {"mean_values": "", "scale_values": ""},
        }


class MockModel:
    class _Configuration:
        def __init__(self, label_schema):
            self.label_schema = label_schema

        def get_label_schema(self):
            return self.label_schema

    def __init__(self):
        self.model_adapters = ["weights.pth"]
        self.data = np.ndarray(1)

        label_schema = generate_otx_label_schema()

        self.configuration = self._Configuration(label_schema)

    def get_data(self, name):
        return self.data

    def set_data(self, *args, **kwargs):
        return


class TestOTXSegmentationTask:
    @pytest.fixture(autouse=True)
    def setup(self):
        model_template = parse_model_template(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        task_env = init_environment(hyper_parameters, model_template)

        self.seg_task = MockOTXSegmentationTask(task_env)

    @e2e_pytest_unit
    def test_load_model_ckpt(self, mocker):
        mocker_torch_load = mocker.patch("torch.load")
        self.seg_task._load_model_ckpt(MockModel())
        mocker_torch_load.assert_called_once()

    @e2e_pytest_unit
    def test_train(self, mocker):
        dataset = generate_otx_dataset(5)
        mocker.patch("torch.load", return_value=np.ndarray([1]))
        self.seg_task.train(dataset, MockModel())
        assert self.seg_task._model_ckpt == "dummy.pth"

    @e2e_pytest_unit
    def test_infer(self):
        dataset = generate_otx_dataset(5)
        predicted_dataset = self.seg_task.infer(
            dataset.with_empty_annotations(), inference_parameters=InferenceParameters(is_evaluation=False)
        )
        assert predicted_dataset[0].annotation_scene.annotations[0]

    @e2e_pytest_unit
    def test_evaluate(self, mocker):
        class _MockScoreMetric:
            def __init__(self, value):
                self.value = value

        class _MockMetric:
            def __init__(self):
                self.overall_dice = _MockScoreMetric(1.0)

            def get_performance(self):
                return 1.0

        class _MockResultEntity:
            performance = 0.0

        mocker.patch(
            "otx.algorithms.segmentation.task.MetricsHelper.compute_dice_averaged_over_pixels",
            return_value=_MockMetric(),
        )

        _result_entity = _MockResultEntity()
        self.seg_task.evaluate(_result_entity)
        assert _result_entity.performance == 1.0

    @e2e_pytest_unit
    @pytest.mark.parametrize("export_type", [ExportType.ONNX, ExportType.OPENVINO])
    def test_export(self, otx_model, mocker, export_type):
        mocker_open = mocker.patch("builtins.open")
        mocker_open.__enter__.return_value = True
        mocker.patch("otx.algorithms.segmentation.task.embed_ir_model_data", return_value=None)
        mocker.patch("otx.algorithms.segmentation.task.embed_onnx_model_data", return_value=None)
        self.seg_task.export(export_type, otx_model)
        mocker_open.assert_called()
