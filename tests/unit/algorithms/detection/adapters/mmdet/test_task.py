"""Unit Test for otx.algorithms.detection.adapters.mmdet.task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from contextlib import nullcontext
from typing import Any, Dict

import numpy as np
from otx.algorithms.common.utils.utils import is_xpu_available
import pytest
import torch
from torch import nn

from otx.algorithms.common.adapters.mmcv.utils import config_utils
from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig
from otx.algorithms.detection.adapters.mmdet.task import MMDetectionTask
from otx.algorithms.detection.adapters.mmdet.models.detectors.custom_atss_detector import CustomATSS
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.api.configuration.helper import create
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.explain_parameters import ExplainParameters
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.model import (
    ModelConfiguration,
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from otx.api.entities.model_template import parse_model_template, TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_TEMPLATE_DIR,
    DEFAULT_ISEG_TEMPLATE_DIR,
    init_environment,
    generate_det_dataset,
)

import pycocotools.mask as mask_util


class MockModule(nn.Module):
    """Mock class for nn.Module."""

    def forward(self, inputs: Any):
        return inputs


class MockModel(nn.Module):
    """Mock class for pytorch model."""

    def __init__(self, task_type):
        super().__init__()
        self.module = MockModule()
        self.module.backbone = MockModule()
        self.backbone = MockModule()
        self.task_type = task_type

    def forward(self, *args, **kwargs):
        forward_hooks = list(self.module.backbone._forward_hooks.values())
        for hook in forward_hooks:
            hook(1, 2, 3)
        return [[np.array([[0, 0, 1, 1, 0.1]]), np.array([[0, 0, 1, 1, 0.2]]), np.array([[0, 0, 1, 1, 0.7]])]]

    @staticmethod
    def named_parameters():
        return {"name": torch.Tensor([0.5])}.items()


class MockDataset(DatasetEntity):
    """Mock class for mm_dataset."""

    def __init__(self, dataset: DatasetEntity, task_type: str):
        self.otx_dataset = dataset
        self.task_type = task_type
        self.CLASSES = ["1", "2", "3"]

    def __len__(self):
        return len(self.otx_dataset)

    def evaluate(self, prediction, *args, **kwargs):
        if self.task_type == "det":
            return {"mAP": 1.0}
        else:
            return {"mAP": 1.0}


class MockDataLoader:
    """Mock class for data loader."""

    def __init__(self, dataset: DatasetEntity):
        self.otx_dataset = dataset
        self.iter = iter(self.otx_dataset)

    def __len__(self) -> int:
        return len(self.otx_dataset)

    def __next__(self) -> Dict[str, DatasetItemEntity]:
        return {"imgs": next(self.iter)}

    def __iter__(self):
        return self


class MockExporter:
    """Mock class for Exporter."""

    def __init__(self, task):
        self._output_path = task._output_path

    def run(self, cfg, *args, **kwargs):
        cfg.model.bbox_head.num_classes == 3
        with open(os.path.join(self._output_path, "openvino.bin"), "wb") as f:
            f.write(np.ndarray([0]))
        with open(os.path.join(self._output_path, "openvino.xml"), "wb") as f:
            f.write(np.ndarray([0]))
        with open(os.path.join(self._output_path, "model.onnx"), "wb") as f:
            f.write(np.ndarray([0]))

        return {
            "outputs": {
                "bin": os.path.join(self._output_path, "openvino.bin"),
                "xml": os.path.join(self._output_path, "openvino.xml"),
                "onnx": os.path.join(self._output_path, "model.onnx"),
            }
        }


class TestMMDetectionTask:
    """Test class for MMDetectionTask.

    Details are explained in each test function.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        model_template = parse_model_template(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.auto_num_workers = True
        task_env = init_environment(hyper_parameters, model_template, task_type=TaskType.DETECTION)

        self.det_task = MMDetectionTask(task_env)

        self.det_dataset, self.det_labels = generate_det_dataset(TaskType.DETECTION, 100)
        self.det_label_schema = LabelSchemaEntity()
        det_label_group = LabelGroup(
            name="labels",
            labels=self.det_labels,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        self.det_label_schema.add_group(det_label_group)

        model_template = parse_model_template(os.path.join(DEFAULT_ISEG_TEMPLATE_DIR, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.auto_adapt_batch_size = "None"
        task_env = init_environment(hyper_parameters, model_template, task_type=TaskType.INSTANCE_SEGMENTATION)

        self.iseg_task = MMDetectionTask(task_env)

        self.iseg_dataset, self.iseg_labels = generate_det_dataset(TaskType.INSTANCE_SEGMENTATION, 100)
        self.iseg_label_schema = LabelSchemaEntity()
        iseg_label_group = LabelGroup(
            name="labels",
            labels=self.iseg_labels,
            group_type=LabelGroupType.EXCLUSIVE,
        )
        self.iseg_label_schema.add_group(iseg_label_group)

    @e2e_pytest_unit
    def test_build_model(self, mocker) -> None:
        """Test build_model function."""
        _mock_recipe_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "model.py"))
        model = self.det_task.build_model(_mock_recipe_cfg, True)
        assert isinstance(model, CustomATSS)

    @e2e_pytest_unit
    def test_load_postprocessing(self):
        """Test _load_postprocessing function."""
        mock_model_data = {
            "config": {
                "postprocessing": {
                    "use_ellipse_shapes": {"value": True},
                    "nms_iou_threshold": {"value": 0.4},
                },
            },
            "confidence_threshold": 0.75,
        }
        self.det_task._load_postprocessing(mock_model_data)
        assert self.det_task._hyperparams.postprocessing.use_ellipse_shapes == True
        assert self.det_task.confidence_threshold == 0.75
        assert self.det_task.nms_iou_threshold == 0.4

        mock_model_data = {
            "config": {
                "postprocessing": {
                    "use_ellipse_shapes": {"value": False},
                    "nms_iou_threshold": {"value": 0.4},
                },
            },
            "confidence_threshold": 0.75,
        }
        self.det_task._hyperparams.postprocessing.result_based_confidence_threshold = False
        self.det_task._hyperparams.postprocessing.confidence_threshold = 0.45
        self.det_task.nms_iou_threshold = 0.45
        self.det_task._load_postprocessing(mock_model_data)
        assert self.det_task._hyperparams.postprocessing.use_ellipse_shapes == False
        assert self.det_task.confidence_threshold == 0.45
        assert self.det_task.nms_iou_threshold == 0.4

    @e2e_pytest_unit
    def test_train(self, mocker) -> None:
        """Test train function."""

        def _mock_train_detector_det(*args, **kwargs):
            with open(os.path.join(self.det_task._output_path, "latest.pth"), "wb") as f:
                torch.save({"dummy": torch.randn(1, 3, 3, 3)}, f)

        def _mock_train_detector_iseg(*args, **kwargs):
            with open(os.path.join(self.iseg_task._output_path, "latest.pth"), "wb") as f:
                torch.save({"dummy": torch.randn(1, 3, 3, 3)}, f)

        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataset",
            return_value=MockDataset(self.det_dataset, "det"),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataloader",
            return_value=MockDataLoader(self.det_dataset),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.train_detector",
            side_effect=_mock_train_detector_det,
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.single_gpu_test",
            return_value=[
                np.array([np.array([[0, 0, 1, 1, 0.1]]), np.array([[0, 0, 1, 1, 0.2]]), np.array([[0, 0, 1, 1, 0.7]])])
            ]
            * 100,
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.FeatureVectorHook",
            return_value=nullcontext(),
        )

        # mock for testing num_workers
        num_cpu = 20
        mock_multiprocessing = mocker.patch.object(config_utils, "multiprocessing")
        mock_multiprocessing.cpu_count.return_value = num_cpu
        num_gpu = 5
        mock_torch = mocker.patch.object(config_utils, "torch")
        mock_torch.cuda.device_count.return_value = num_gpu
        if is_xpu_available():
            mock_devcnt = mocker.patch.object(config_utils, "get_adaptive_num_workers")
            mock_devcnt.return_value = num_cpu // num_gpu

        _config = ModelConfiguration(DetectionConfig(), self.det_label_schema)
        output_model = ModelEntity(self.det_dataset, _config)
        self.det_task.train(self.det_dataset, output_model)
        output_model.performance == 1.0
        assert (
            self.det_task._config.data.train_dataloader.workers_per_gpu == num_cpu // num_gpu
        )  # test adaptive num_workers

        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.train_detector",
            side_effect=_mock_train_detector_iseg,
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.single_gpu_test",
            return_value=[(np.array([[[0, 0, 1, 1, 1]]]), np.ones((1, 1, 28, 28)))] * 100,
        )
        _config = ModelConfiguration(DetectionConfig(), self.iseg_label_schema)
        output_model = ModelEntity(self.iseg_dataset, _config)
        self.iseg_task.train(self.iseg_dataset, output_model)
        output_model.performance == 1.0
        assert (
            self.det_task._config.data.train_dataloader.workers_per_gpu == num_cpu // num_gpu
        )  # test adaptive num_workers

        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.single_gpu_test",
            return_value=[
                (
                    np.array([[[0, 0, 1, 1, 1]]]),
                    [[mask_util.encode(np.ones((28, 28, 1), dtype=np.uint8, order="F"))[0]]],
                )
            ]
            * 100,
        )
        _config = ModelConfiguration(DetectionConfig(), self.iseg_label_schema)
        output_model = ModelEntity(self.iseg_dataset, _config)
        self.iseg_task.train(self.iseg_dataset, output_model)
        output_model.performance == 1.0
        assert (
            self.det_task._config.data.train_dataloader.workers_per_gpu == num_cpu // num_gpu
        )  # test adaptive num_workers

    @e2e_pytest_unit
    def test_infer(self, mocker) -> None:
        """Test infer function."""

        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataset",
            return_value=MockDataset(self.det_dataset, "det"),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataloader",
            return_value=MockDataLoader(self.det_dataset),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.single_gpu_test",
            return_value=[
                np.array([np.array([[0, 0, 1, 1, 0.1]]), np.array([[0, 0, 1, 1, 0.2]]), np.array([[0, 0, 1, 1, 0.7]])])
            ]
            * 100,
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.FeatureVectorHook",
            return_value=nullcontext(),
        )

        inference_parameters = InferenceParameters(is_evaluation=True)
        outputs = self.det_task.infer(self.det_dataset, inference_parameters)
        for output in outputs:
            assert output.get_annotations()[-1].get_labels()[0].probability == 0.7

    @e2e_pytest_unit
    def test_det_evaluate(self) -> None:
        """Test evaluate function for detection."""

        _config = ModelConfiguration(DetectionConfig(), self.det_label_schema)
        _model = ModelEntity(self.det_dataset, _config)
        resultset = ResultSetEntity(_model, self.det_dataset, self.det_dataset)
        self.det_task.evaluate(resultset)
        assert resultset.performance.score.value == 1.0

    @e2e_pytest_unit
    def test_det_evaluate_with_empty_annotations(self) -> None:
        """Test evaluate function for detection with empty predictions."""

        _config = ModelConfiguration(DetectionConfig(), self.det_label_schema)
        _model = ModelEntity(self.det_dataset, _config)
        resultset = ResultSetEntity(_model, self.det_dataset, self.det_dataset.with_empty_annotations())
        self.det_task.evaluate(resultset)
        assert resultset.performance.score.value == 0.0

    @e2e_pytest_unit
    def test_iseg_evaluate(self) -> None:
        """Test evaluate function for instance segmentation."""

        _config = ModelConfiguration(DetectionConfig(), self.iseg_label_schema)
        _model = ModelEntity(self.iseg_dataset, _config)
        resultset = ResultSetEntity(_model, self.iseg_dataset, self.iseg_dataset)
        self.iseg_task.evaluate(resultset)
        assert resultset.performance.score.value == 1.0

    @pytest.mark.parametrize("precision", [ModelPrecision.FP16, ModelPrecision.FP32])
    @e2e_pytest_unit
    def test_export(self, mocker, precision: ModelPrecision, export_type: ExportType = ExportType.OPENVINO) -> None:
        """Test export function.

        <Steps>
            1. Create model entity
            2. Run export function
            3. Check output model attributes
        """
        _config = ModelConfiguration(DetectionConfig(), self.det_label_schema)
        _model = ModelEntity(self.det_dataset, _config)

        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.DetectionExporter",
            return_value=MockExporter(self.det_task),
        )
        mocker.patch(
            "otx.algorithms.detection.task.embed_ir_model_data",
            return_value=True,
        )

        self.det_task.export(export_type, _model, precision, False)

        assert _model.model_format == ModelFormat.ONNX if export_type == ExportType.ONNX else ModelFormat.OPENVINO
        assert _model.precision[0] == precision
        assert _model.precision == self.det_task._precision

        if export_type == ExportType.OPENVINO:
            assert _model.get_data("openvino.bin") is not None
            assert _model.get_data("openvino.xml") is not None
            assert _model.optimization_type == ModelOptimizationType.MO
        else:
            assert _model.get_data("model.onnx") is not None
            assert _model.optimization_type == ModelOptimizationType.ONNX

        assert _model.get_data("confidence_threshold") is not None
        assert _model.optimization_methods == self.det_task._optimization_methods
        assert _model.get_data("label_schema.json") is not None

    @e2e_pytest_unit
    def test_export_onnx(self, mocker) -> None:
        self.test_export(mocker, ModelPrecision.FP32, ExportType.ONNX)

    @e2e_pytest_unit
    def test_explain(self, mocker):
        """Test explain function."""

        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataset",
            return_value=MockDataset(self.det_dataset, "det"),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataloader",
            return_value=MockDataLoader(self.det_dataset),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_data_parallel",
            return_value=MockModel(TaskType.DETECTION),
        )

        explain_parameters = ExplainParameters(
            explainer="ClassWiseSaliencyMap",
            process_saliency_maps=False,
            explain_predicted_classes=True,
        )
        outputs = self.det_task.explain(self.det_dataset, explain_parameters)

    @e2e_pytest_unit
    def test_anchor_clustering(self, mocker):

        ssd_dir = os.path.join("src/otx/algorithms/detection/configs/detection", "mobilenetv2_ssd")
        ssd_cfg = OTXConfig.fromfile(os.path.join(ssd_dir, "model.py"))
        model_template = parse_model_template(os.path.join(ssd_dir, "template.yaml"))
        hyper_parameters = create(model_template.hyper_parameters.data)
        hyper_parameters.learning_parameters.auto_num_workers = True
        task_env = init_environment(hyper_parameters, model_template, task_type=TaskType.DETECTION)

        det_task = MMDetectionTask(task_env)

        def _mock_train_detector_det(*args, **kwargs):
            with open(os.path.join(self.det_task._output_path, "latest.pth"), "wb") as f:
                torch.save({"dummy": torch.randn(1, 3, 3, 3)}, f)

        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataset",
            return_value=MockDataset(self.det_dataset, "det"),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataloader",
            return_value=MockDataLoader(self.det_dataset),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.train_detector",
            side_effect=_mock_train_detector_det,
        )

        det_task._train_model(self.det_dataset)
        assert ssd_cfg.model.bbox_head.anchor_generator != det_task.config.model.bbox_head.anchor_generator

        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.single_gpu_test",
            return_value=[
                np.array([np.array([[0, 0, 1, 1, 0.1]]), np.array([[0, 0, 1, 1, 0.2]]), np.array([[0, 0, 1, 1, 0.7]])])
            ]
            * 100,
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.FeatureVectorHook",
            return_value=nullcontext(),
        )
        inference_parameters = InferenceParameters(is_evaluation=True)
        det_task._infer_model(self.det_dataset, inference_parameters)
        assert ssd_cfg.model.bbox_head.anchor_generator != det_task.config.model.bbox_head.anchor_generator

    @e2e_pytest_unit
    def test_geti_scenario(self, mocker):
        """Test Geti scenario.

        Train -> Eval -> Export
        """

        # TRAIN
        def _mock_train_detector_det(*args, **kwargs):
            with open(os.path.join(self.det_task._output_path, "latest.pth"), "wb") as f:
                torch.save({"dummy": torch.randn(1, 3, 3, 3)}, f)

        def _mock_train_detector_iseg(*args, **kwargs):
            with open(os.path.join(self.iseg_task._output_path, "latest.pth"), "wb") as f:
                torch.save({"dummy": torch.randn(1, 3, 3, 3)}, f)

        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataset",
            return_value=MockDataset(self.det_dataset, "det"),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataloader",
            return_value=MockDataLoader(self.det_dataset),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.train_detector",
            side_effect=_mock_train_detector_det,
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.single_gpu_test",
            return_value=[
                np.array([np.array([[0, 0, 1, 1, 0.1]]), np.array([[0, 0, 1, 1, 0.2]]), np.array([[0, 0, 1, 1, 0.7]])])
            ]
            * 100,
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.FeatureVectorHook",
            return_value=nullcontext(),
        )

        # mock for testing num_workers
        num_cpu = 20
        mock_multiprocessing = mocker.patch.object(config_utils, "multiprocessing")
        mock_multiprocessing.cpu_count.return_value = num_cpu
        num_gpu = 5
        mock_torch = mocker.patch.object(config_utils, "torch")
        mock_torch.cuda.device_count.return_value = num_gpu
        if is_xpu_available():
            mock_devcnt = mocker.patch.object(config_utils, "get_adaptive_num_workers")
            mock_devcnt.return_value = 1

        _config = ModelConfiguration(DetectionConfig(), self.det_label_schema)
        output_model = ModelEntity(self.det_dataset, _config)
        self.det_task.train(self.det_dataset, output_model)

        # INFER
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataset",
            return_value=MockDataset(self.det_dataset, "det"),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.build_dataloader",
            return_value=MockDataLoader(self.det_dataset),
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.single_gpu_test",
            return_value=[
                np.array([np.array([[0, 0, 1, 1, 0.1]]), np.array([[0, 0, 1, 1, 0.2]]), np.array([[0, 0, 1, 1, 0.7]])])
            ]
            * 100,
        )
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.FeatureVectorHook",
            return_value=nullcontext(),
        )

        inference_parameters = InferenceParameters(is_evaluation=True)
        outputs = self.det_task.infer(self.det_dataset, inference_parameters)

        # EXPORT
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.task.DetectionExporter",
            return_value=MockExporter(self.det_task),
        )
        mocker.patch(
            "otx.algorithms.detection.task.embed_ir_model_data",
            return_value=True,
        )

        export_type = ExportType.OPENVINO
        precision = ModelPrecision.FP32
        self.det_task.export(export_type, output_model, precision, False)
