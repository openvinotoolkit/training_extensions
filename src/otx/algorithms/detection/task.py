"""Task of OTX Detection."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import psutil
import torch
from mmcv.utils import ConfigDict

from otx.algorithms.common.configs.configuration_enums import InputSizePreset
from otx.algorithms.common.tasks.base_task import TRAIN_TYPE_DIR_PATH, OTXTask
from otx.algorithms.common.utils.callback import (
    InferenceProgressCallback,
    TrainingProgressCallback,
)
from otx.algorithms.common.utils.ir import embed_ir_model_data
from otx.algorithms.common.utils.utils import embed_onnx_model_data, get_cfg_based_on_device
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.utils import create_detection_shapes, create_mask_shapes, get_det_model_api_configuration
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import config_to_bytes, ids_to_strings
from otx.api.entities.datasets import DatasetEntity, DatasetPurpose
from otx.api.entities.explain_parameters import ExplainParameters
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.metrics import (
    BarChartInfo,
    BarMetricsGroup,
    CurveMetric,
    LineChartInfo,
    LineMetricsGroup,
    MetricsGroup,
    ScoreMetric,
    VisualizationType,
)
from otx.api.entities.model import (
    ModelEntity,
    ModelPrecision,
)
from otx.api.entities.model_template import TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.entities.train_parameters import TrainParameters, default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.utils.dataset_utils import add_saliency_maps_to_dataset_item
from otx.cli.utils.multi_gpu import is_multigpu_child_process
from otx.core.data.caching.mem_cache_handler import MemCacheHandlerSingleton
from otx.utils.logger import get_logger

logger = get_logger()


class OTXDetectionTask(OTXTask, ABC):
    """Task class for OTX detection."""

    # pylint: disable=too-many-instance-attributes, too-many-locals
    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__(task_environment, output_path)
        self._task_config = DetectionConfig
        self._hyperparams: ConfigDict = task_environment.get_hyper_parameters(self._task_config)
        self._train_type = self._hyperparams.algo_backend.train_type
        self._model_dir = os.path.join(
            os.path.abspath(os.path.dirname(self._task_environment.model_template.model_template_path)),
            TRAIN_TYPE_DIR_PATH[self._train_type.name],
        )
        self._anchors: Dict[str, int] = {}

        self.confidence_threshold = 0.0
        self.max_num_detections = 0
        if hasattr(self._hyperparams, "postprocessing"):
            if hasattr(self._hyperparams.postprocessing, "confidence_threshold"):
                self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
            if hasattr(self._hyperparams.postprocessing, "max_num_detections"):
                self.max_num_detections = self._hyperparams.postprocessing.max_num_detections

        if task_environment.model is not None:
            self._load_model()

        self.use_ellipse_shapes = self._hyperparams.postprocessing.use_ellipse_shapes

        if self._hyperparams.tiling_parameters.enable_tiling:
            self.data_pipeline_path = os.path.join(self._model_dir, "tile_pipeline.py")
        else:
            self.data_pipeline_path = get_cfg_based_on_device(os.path.join(self._model_dir, "data_pipeline.py"))

        if hasattr(self._hyperparams.learning_parameters, "input_size"):
            input_size_cfg = InputSizePreset(self._hyperparams.learning_parameters.input_size.value)
        else:
            input_size_cfg = InputSizePreset.DEFAULT
        if self._hyperparams.tiling_parameters.enable_tiling:
            # Disable auto input size if tiling is enabled
            input_size_cfg = InputSizePreset.DEFAULT
        self._input_size = input_size_cfg.tuple

    def _load_postprocessing(self, model_data):
        """Load postprocessing configs form PyTorch model.

        Args:
            model_data: The model data.
        """
        loaded_postprocessing = model_data.get("config", {}).get("postprocessing", {})
        hparams = self._hyperparams.postprocessing
        if "use_ellipse_shapes" in loaded_postprocessing:
            hparams.use_ellipse_shapes = loaded_postprocessing["use_ellipse_shapes"]["value"]
        else:
            hparams.use_ellipse_shapes = False
        if "max_num_detections" in loaded_postprocessing:
            trained_max_num_detections = loaded_postprocessing["max_num_detections"]["value"]
            # Prefer new hparam value set by user (>0) intentionally than trained value
            if self.max_num_detections == 0:
                self.max_num_detections = trained_max_num_detections

        # If confidence threshold is adaptive then up-to-date value should be stored in the model
        # and should not be changed during inference. Otherwise user-specified value should be taken.
        if hparams.result_based_confidence_threshold:
            self.confidence_threshold = model_data.get("confidence_threshold", self.confidence_threshold)
        else:
            self.confidence_threshold = hparams.confidence_threshold
        logger.info(f"Confidence threshold {self.confidence_threshold}")

    def _load_tiling_parameters(self, model_data):
        """Load tiling parameters from PyTorch model.

        Args:
            model_data: The model data.

        Raises:
            RuntimeError: If tile classifier is enabled but not found in the trained model.
        """
        loaded_tiling_parameters = model_data.get("config", {}).get("tiling_parameters", {})
        if loaded_tiling_parameters.get("enable_tiling", {}).get("value", False):
            logger.info("Load tiling parameters")
            hparams = self._hyperparams.tiling_parameters
            hparams.enable_tiling = loaded_tiling_parameters["enable_tiling"]["value"]
            hparams.tile_size = loaded_tiling_parameters["tile_size"]["value"]
            hparams.tile_overlap = loaded_tiling_parameters["tile_overlap"]["value"]
            hparams.tile_max_number = loaded_tiling_parameters["tile_max_number"]["value"]
            hparams.tile_ir_scale_factor = loaded_tiling_parameters["tile_ir_scale_factor"]["value"]
            hparams.object_tile_ratio = loaded_tiling_parameters["object_tile_ratio"]["value"]
            # check backward compatibility
            enable_tile_classifier = loaded_tiling_parameters.get("enable_tile_classifier", {}).get("value", False)
            if enable_tile_classifier:
                found_tile_classifier = any(
                    layer_name.startswith("tile_classifier") for layer_name in model_data["model"]["state_dict"].keys()
                )
                if not found_tile_classifier:
                    raise RuntimeError(
                        "Tile classifier is enabled but not found in the trained model. Please retrain your model."
                    )
                hparams.enable_tile_classifier = loaded_tiling_parameters["enable_tile_classifier"]["value"]

    def _load_model_ckpt(self, model: Optional[ModelEntity]) -> Optional[Dict]:
        """Load model checkpoint from model entity.

        Args:
            model (Optional[ModelEntity]): The model entity.

        Returns:
            dict: The model checkpoint including model weights and other parameters.
        """
        if model and "weights.pth" in model.model_adapters:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))

            if model_data.get("anchors"):
                self._anchors = model_data["anchors"]
            self._load_postprocessing(model_data)
            self._load_tiling_parameters(model_data)
            return model_data
        return None

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: Optional[TrainParameters] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        """Train function for OTX detection task.

        Actual training is processed by _train_model fucntion
        """
        logger.info("train()")
        logger.info(f"------> system virtual mem: {psutil.virtual_memory()}")
        # Check for stop signal when training has stopped.
        # If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return
        self.seed = seed
        self.deterministic = deterministic

        # Set OTX LoggerHook & Time Monitor
        if train_parameters:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback
        self._time_monitor = TrainingProgressCallback(update_progress_callback)

        dataset.purpose = DatasetPurpose.TRAINING
        results = self._train_model(dataset)

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

        # get output model
        model_ckpt = results.get("final_ckpt")
        if model_ckpt is None:
            logger.error("cannot find final checkpoint from the results.")
            # output_model.model_status = ModelStatus.FAILED
            return
        # update checkpoint to the newly trained model
        self._model_ckpt = model_ckpt

        # get prediction on validation set
        self._is_training = False
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        val_dataset.purpose = DatasetPurpose.INFERENCE
        val_preds, val_map = self._infer_model(val_dataset, InferenceParameters(is_evaluation=True))

        MemCacheHandlerSingleton.delete()

        preds_val_dataset = val_dataset.with_empty_annotations()
        if self._hyperparams.postprocessing.result_based_confidence_threshold:
            confidence_threshold = 0.0  # Use all predictions to compute best threshold
        else:
            confidence_threshold = self.confidence_threshold
        self._add_predictions_to_dataset(
            val_preds,
            preds_val_dataset,
            confidence_threshold=confidence_threshold,
            use_ellipse_shapes=self.use_ellipse_shapes,
        )

        result_set = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=val_dataset,
            prediction_dataset=preds_val_dataset,
        )

        # adjust confidence threshold
        if self._hyperparams.postprocessing.result_based_confidence_threshold:
            best_confidence_threshold = None
            logger.info("Adjusting the confidence threshold")
            metric = MetricsHelper.compute_f_measure(result_set, vary_confidence_threshold=True)
            if metric.best_confidence_threshold:
                best_confidence_threshold = metric.best_confidence_threshold.value
            if best_confidence_threshold is None:
                raise ValueError("Cannot compute metrics: Invalid confidence threshold!")
            logger.info(f"Setting confidence threshold to {best_confidence_threshold} based on results")
            self.confidence_threshold = best_confidence_threshold
        else:
            metric = MetricsHelper.compute_f_measure(result_set, vary_confidence_threshold=False)

        # compose performance statistics
        # TODO[EUGENE]: HOW TO ADD A MAE CURVE FOR TaskType.COUNTING?
        performance = metric.get_performance()
        performance.dashboard_metrics.extend(self._generate_training_metrics(self._learning_curves, val_map))
        logger.info(f"Final model performance: {str(performance)}")
        # save resulting model
        self.save_model(output_model)
        output_model.performance = performance
        logger.info("train done.")

    @abstractmethod
    def _train_model(self, dataset: DatasetEntity):
        """Train model and return the results."""
        raise NotImplementedError

    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main infer function."""
        logger.info("infer()")
        process_saliency_maps = False
        explain_predicted_classes = True

        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress  # type: ignore
            process_saliency_maps = inference_parameters.process_saliency_maps
            explain_predicted_classes = inference_parameters.explain_predicted_classes

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        dataset.purpose = DatasetPurpose.INFERENCE
        prediction_results, _ = self._infer_model(dataset, inference_parameters)
        self._add_predictions_to_dataset(
            prediction_results,
            dataset,
            self.confidence_threshold,
            process_saliency_maps,
            explain_predicted_classes,
            self.use_ellipse_shapes,
        )
        logger.info("Inference completed")
        return dataset

    @abstractmethod
    def _infer_model(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ):
        """Get inference results from dataset."""
        raise NotImplementedError

    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = True,
    ):
        """Export function of OTX Detection Task."""
        logger.info("Exporting the model")

        self._update_model_export_metadata(output_model, export_type, precision, dump_features)

        results = self._export_model(precision, export_type, dump_features)
        outputs = results.get("outputs")
        logger.debug(f"results of run_task = {outputs}")
        if outputs is None:
            raise RuntimeError(results.get("msg"))

        ir_extra_data = get_det_model_api_configuration(
            self._task_environment.label_schema,
            self._task_type,
            self.confidence_threshold,
            self._hyperparams.tiling_parameters,
            self.use_ellipse_shapes,
        )

        if export_type == ExportType.ONNX:
            ir_extra_data[("model_info", "mean_values")] = results.get("inference_parameters").get("mean_values")
            ir_extra_data[("model_info", "scale_values")] = results.get("inference_parameters").get("scale_values")

            onnx_file = outputs.get("onnx")
            embed_onnx_model_data(onnx_file, ir_extra_data)
            with open(onnx_file, "rb") as f:
                output_model.set_data("model.onnx", f.read())
        else:
            bin_file = outputs.get("bin")
            xml_file = outputs.get("xml")

            embed_ir_model_data(xml_file, ir_extra_data)

            with open(bin_file, "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            with open(xml_file, "rb") as f:
                output_model.set_data("openvino.xml", f.read())

        if self._hyperparams.tiling_parameters.enable_tile_classifier:
            tile_classifier = None
            for partition in outputs.get("partitioned", {}):
                if partition.get("tile_classifier"):
                    tile_classifier = partition.get("tile_classifier")
                    break
            if tile_classifier is None:
                raise RuntimeError("invalid status of exporting. tile_classifier should not be None")
            if export_type == ExportType.ONNX:
                with open(tile_classifier["onnx"], "rb") as f:
                    output_model.set_data("tile_classifier.onnx", f.read())
            else:
                with open(tile_classifier["bin"], "rb") as f:
                    output_model.set_data("tile_classifier.bin", f.read())
                with open(tile_classifier["xml"], "rb") as f:
                    output_model.set_data("tile_classifier.xml", f.read())

        output_model.set_data(
            "confidence_threshold",
            np.array([self.confidence_threshold], dtype=np.float32).tobytes(),
        )
        output_model.set_data("config.json", config_to_bytes(self._hyperparams))
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        logger.info("Exporting completed")

    @abstractmethod
    def _export_model(self, precision: ModelPrecision, export_format: ExportType, dump_features: bool):
        """Main export function using training backend."""
        raise NotImplementedError

    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[ExplainParameters] = None,
    ) -> DatasetEntity:
        """Main explain function of OTX Task."""
        logger.info("explain()")

        update_progress_callback = default_progress_callback
        process_saliency_maps = False
        explain_predicted_classes = True
        if explain_parameters is not None:
            update_progress_callback = explain_parameters.update_progress  # type: ignore
            process_saliency_maps = explain_parameters.process_saliency_maps
            explain_predicted_classes = explain_parameters.explain_predicted_classes

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        outputs = self._explain_model(dataset, explain_parameters)
        detections = outputs["detections"]
        explain_results = outputs["saliency_maps"]

        self._add_explanations_to_dataset(
            detections, explain_results, dataset, process_saliency_maps, explain_predicted_classes
        )
        logger.info("Explain completed")
        return dataset

    @abstractmethod
    def _explain_model(self, dataset: DatasetEntity, explain_parameters: Optional[ExplainParameters]):
        raise NotImplementedError

    def evaluate(
        self,
        output_resultset: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        """Evaluate function of OTX Detection Task."""
        logger.info("called evaluate()")
        if evaluation_metric is not None:
            logger.warning(
                f"Requested to use {evaluation_metric} metric, " "but parameter is ignored. Use F-measure instead."
            )
        metric = MetricsHelper.compute_f_measure(output_resultset)
        output_resultset.performance = metric.get_performance()
        logger.info(f"F-measure after evaluation: {output_resultset.performance}")
        logger.info("Evaluation completed")

    def _add_predictions_to_dataset(
        self,
        prediction_results,
        dataset,
        confidence_threshold=0.0,
        process_saliency_maps=False,
        explain_predicted_classes=True,
        use_ellipse_shapes=False,
    ):
        """Loop over dataset again to assign predictions. Convert from MMDetection format to OTX format."""
        for dataset_item, (all_results, feature_vector, saliency_map) in zip(dataset, prediction_results):
            shapes = self._get_shapes(
                all_results, dataset_item.width, dataset_item.height, confidence_threshold, use_ellipse_shapes
            )
            dataset_item.append_annotations(shapes)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if saliency_map is not None:
                labels = self._labels.copy()
                if len(saliency_map) == len(labels) + 1:
                    # Include the background as the last category
                    labels.append(LabelEntity("background", Domain.DETECTION))

                predicted_scored_labels = []
                for shape in shapes:
                    predicted_scored_labels += shape.get_labels()

                add_saliency_maps_to_dataset_item(
                    dataset_item=dataset_item,
                    saliency_map=saliency_map,
                    model=self._task_environment.model,
                    labels=labels,
                    predicted_scored_labels=predicted_scored_labels,
                    explain_predicted_classes=explain_predicted_classes,
                    process_saliency_maps=process_saliency_maps,
                )

    def _get_shapes(self, all_results, width, height, confidence_threshold, use_ellipse_shapes):
        if self._task_type == TaskType.DETECTION:
            shapes = create_detection_shapes(
                all_results, width, height, confidence_threshold, use_ellipse_shapes, self._labels
            )
        elif self._task_type in {
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.ROTATED_DETECTION,
        }:
            shapes = create_mask_shapes(
                all_results,
                width,
                height,
                confidence_threshold,
                use_ellipse_shapes,
                self._labels,
                self._task_type is TaskType.ROTATED_DETECTION,
            )
        else:
            raise RuntimeError(f"OTX results assignment not implemented for task: {self._task_type}")
        return shapes

    def _add_explanations_to_dataset(
        self,
        detections,
        explain_results,
        dataset,
        process_saliency_maps,
        explain_predicted_classes,
        use_ellipse_shapes=False,
    ):
        """Add saliency map to the dataset."""
        for dataset_item, detection, saliency_map in zip(dataset, detections, explain_results):
            labels = self._labels.copy()
            if len(saliency_map) == len(labels) + 1:
                # Include the background as the last category
                labels.append(LabelEntity("background", Domain.DETECTION))

            shapes = self._get_shapes(
                detection, dataset_item.width, dataset_item.height, self.confidence_threshold, use_ellipse_shapes
            )
            predicted_scored_labels = []
            for shape in shapes:
                predicted_scored_labels += shape.get_labels()

            add_saliency_maps_to_dataset_item(
                dataset_item=dataset_item,
                saliency_map=saliency_map,
                model=self._task_environment.model,
                labels=labels,
                predicted_scored_labels=predicted_scored_labels,
                explain_predicted_classes=explain_predicted_classes,
                process_saliency_maps=process_saliency_maps,
            )

    @staticmethod
    def _generate_training_metrics(learning_curves, scores) -> Iterable[MetricsGroup[Any, Any]]:
        """Get Training metrics (epochs & scores).

        Parses the mmdetection logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        # Learning curves.
        for key, curve in learning_curves.items():
            len_x, len_y = len(curve.x), len(curve.y)
            if len_x != len_y:
                logger.warning(f"Learning curve {key} has inconsistent number of coordinates ({len_x} vs {len_y}.")
                len_x = min(len_x, len_y)
                curve.x = curve.x[:len_x]
                curve.y = curve.y[:len_x]
            metric_curve = CurveMetric(
                xs=np.nan_to_num(curve.x).tolist(),
                ys=np.nan_to_num(curve.y).tolist(),
                name=key,
            )
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(LineMetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        # Final mAP value on the validation set.
        output.append(
            BarMetricsGroup(
                metrics=[ScoreMetric(value=scores, name="mAP")],
                visualization_info=BarChartInfo("Validation score", visualization_type=VisualizationType.RADIAL_BAR),
            )
        )

        return output

    def save_model(self, output_model: ModelEntity):
        """Save best model weights in DetectionTrainTask."""
        if is_multigpu_child_process():
            return

        logger.info("called save_model")
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {
            "model": model_ckpt,
            "config": hyperparams_str,
            "labels": labels,
            "confidence_threshold": self.confidence_threshold,
            "input_size": self._input_size,
            "VERSION": 1,
        }
        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        output_model.precision = self._precision
