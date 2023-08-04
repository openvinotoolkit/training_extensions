import io
from typing import List, Optional, Union

import torch
from anomalib.models import AnomalyModule

# OTX V1 API
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.metrics import NullPerformance, Performance, ScoreMetric
from otx.api.entities.model import ModelEntity, ModelFormat, ModelOptimizationType, ModelPrecision, OptimizationMethod
from otx.api.entities.model_template import TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import TrainParameters
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.evaluation.performance_provider_interface import (
    IPerformanceProvider,
)
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.v2.adapters.torch.anomalib import Dataset, get_model

# OTX V2 API
from otx.v2.api.core import AutoRunner

# Pseudo GetiTask
from .pseudo_geti_task import GetiTask


class GetiAnomalibTask(GetiTask):
    """Base Anomaly Task."""

    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None) -> None:
        # Attributes
        self._hyperparams = task_environment.get_hyper_parameters()
        self.task_type = task_environment.model_template.task_type
        self.train_type = self._hyperparams.algo_backend.train_type
        self.precision = [ModelPrecision.FP32]
        self.optimization_methods: List[OptimizationMethod] = []
        self.torch_checkpoint = None

        # Dataset API
        self.label_schema = task_environment.label_schema
        # Model API
        self.model = self.load_model(model=task_environment.model)

        # AutoRunner Settings
        self.auto_runner = AutoRunner(
            framework="anomalib",
            task=self.task_type,
            train_type=self.train_type,
            work_dir=output_path,
        )

    def load_model(self, model: Optional[ModelEntity]) -> Union[torch.nn.Module, AnomalyModule]:
        """Create and Load Anomalib Module from ModelEntity.

        This method checks if the task environment has a saved OTX Model,
        and creates one. If the ModelEntity already exists, it returns the
        the model with the saved weights.

        Args:
            model (Optional[ModelEntity]): ModelEntity from the
                task environment.

        Returns:
            Union[torch.nn.Module, AnomalyModule]: Anomalib
                classification or segmentation model with/without weights.
        """
        if model is None:
            model = get_model(config=self.config)
        else:
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))

            if model_data["config"]["model"]["backbone"] != self.config["model"]["backbone"]:
                self.config["model"]["backbone"] = model_data["config"]["model"]["backbone"]
            try:
                model = get_model(config=self.config)
                model.load_state_dict(model_data["model"])
            except BaseException as exception:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") from exception

        return model

    def dataset_from_entity(self, dataset_entity: DatasetEntity) -> Dataset:
        dataset = Dataset()
        dataset.dataset_entity = dataset_entity
        dataset.initialize = True
        dataset.label_schema = self.label_schema
        return dataset

    def save_model(self, output_model: ModelEntity) -> None:
        """Save the model after training is completed.

        Args:
            output_model (ModelEntity): Output model onto which the weights are saved.
        """
        model_info = {
            "model": self.model.state_dict(),
            "config": self.model.configs,
            "VERSION": 1,
        }

        buffer = io.BytesIO()
        torch.save(model_info, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))

        if hasattr(self.model, "image_metrics"):
            f1_score = self.model.image_metrics.F1Score.compute().item()
            output_model.performance = Performance(score=ScoreMetric(name="F1 Score", value=f1_score))
        else:
            output_model.performance = NullPerformance()
        output_model.precision = self.precision
        output_model.optimization_methods = self.optimization_methods

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        # Covert DatasetEntity -> Dataset
        dataset = self.dataset_from_entity(dataset)
        train_dataloader = dataset.train_dataloader()
        val_dataloader = dataset.val_dataloader()

        # Covert train_parameters -> params
        params = GetiTask.covert_parameter(train_parameters)

        # Update Model
        results = self.auto_runner.train(
            model=self.model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            seed=seed,
            deterministic=deterministic,
            **params,
        )
        self.model = results["model"]
        self.torch_checkpoint = results["checkpoint"]

        # Update training result to ModelEntity
        self.save_model(output_model)

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None) -> None:
        metric: IPerformanceProvider
        if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            metric = MetricsHelper.compute_f_measure(output_resultset)
        elif self.task_type == TaskType.ANOMALY_DETECTION:
            metric = MetricsHelper.compute_anomaly_detection_scores(output_resultset)
        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            metric = MetricsHelper.compute_anomaly_segmentation_scores(output_resultset)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        output_resultset.performance = metric.get_performance()

    def infer(self, dataset_entity: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        # Covert DatasetEntity -> Dataset
        dataset = self.dataset_from_entity(dataset_entity)
        dataloader = dataset.predict_dataloader()

        # Covert train_parameters -> params
        params = GetiTask.covert_parameter(inference_parameters)

        # Update Model
        results = self.auto_runner.predict(model=self.model, img=dataloader, **params)

        # Update result to DatasetEntity
        dataset_entity = GetiTask.get_dataset_entity_from_result(dataset_entity, results)
        return dataset_entity

    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = False,
    ) -> None:
        # Set output_model settings
        if export_type == ExportType.ONNX:
            output_model.model_format = ModelFormat.ONNX
            output_model.optimization_type = ModelOptimizationType.ONNX
            if precision == ModelPrecision.FP16:
                raise RuntimeError("Export to FP16 ONNX is not supported")
        elif export_type == ExportType.OPENVINO:
            output_model.model_format = ModelFormat.OPENVINO
            output_model.optimization_type = ModelOptimizationType.MO
        else:
            raise RuntimeError(f"not supported export type {export_type}")
        self.precision[0] = precision
        output_model.has_xai = dump_features

        # Run Exporting with AutoRunner
        results = self.auto_runner.export(model=self.model)

        # Update output_model with results
        if export_type == ExportType.ONNX:
            onnx_file = results["outputs"]["onnx"]
            with open(onnx_file, "rb") as file:
                output_model.set_data("model.onnx", file.read())
        else:
            bin_file = results["outputs"]["bin"]
            xml_file = results["outputs"]["xml"]
            with open(bin_file, "rb") as file:
                output_model.set_data("openvino.bin", file.read())
            with open(xml_file, "rb") as file:
                output_model.set_data("openvino.xml", file.read())
        output_model.precision = self.precision
        output_model.optimization_methods = self.optimization_methods

        output_model.set_data("label_schema.json", label_schema_to_bytes(self.label_schema))
