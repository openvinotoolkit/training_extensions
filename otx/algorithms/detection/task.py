"""Task of OTX Detection."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import io
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from mmcv.utils import ConfigDict

from otx.algorithms.common.tasks.base_task import TRAIN_TYPE_DIR_PATH, OTXTask
from otx.algorithms.common.utils.callback import (
    InferenceProgressCallback,
    TrainingProgressCallback,
)
from otx.algorithms.common.utils.ir import embed_ir_model_data
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.utils import get_det_model_api_configuration
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import config_to_bytes, ids_to_strings
from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity, DatasetPurpose
from otx.api.entities.explain_parameters import ExplainParameters
from otx.api.entities.id import ID
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
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from otx.api.entities.model_template import TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.entities.train_parameters import TrainParameters, default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.utils.dataset_utils import add_saliency_maps_to_dataset_item
from otx.cli.utils.multi_gpu import is_multigpu_child_process

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

        if hasattr(self._hyperparams, "postprocessing") and hasattr(
            self._hyperparams.postprocessing, "confidence_threshold"
        ):
            self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
        else:
            self.confidence_threshold = 0.0

        if task_environment.model is not None:
            self._load_model()

        if self._hyperparams.tiling_parameters.enable_tiling:
            self.data_pipeline_path = os.path.join(self._model_dir, "tile_pipeline.py")
        else:
            self.data_pipeline_path = os.path.join(self._model_dir, "data_pipeline.py")

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

            # set confidence_threshold as well
            self.confidence_threshold = model_data.get("confidence_threshold", self.confidence_threshold)
            if model_data.get("anchors"):
                self._anchors = model_data["anchors"]
            self._load_tiling_parameters(model_data)
            return model_data
        return None

    def train(
        self, dataset: DatasetEntity, output_model: ModelEntity, train_parameters: Optional[TrainParameters] = None
    ):
        """Train function for OTX detection task.

        Actual training is processed by _train_model fucntion
        """
        logger.info("train()")
        # Check for stop signal when training has stopped.
        # If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

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
        val_preds, val_map = self._infer_model(val_dataset, InferenceParameters(is_evaluation=True))

        preds_val_dataset = val_dataset.with_empty_annotations()
        self._add_predictions_to_dataset(val_preds, preds_val_dataset, 0.0)

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
        # If confidence threshold is adaptive then up-to-date value should be stored in the model
        # and should not be changed during inference. Otherwise user-specified value should be taken.
        if not self._hyperparams.postprocessing.result_based_confidence_threshold:
            self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
        logger.info(f"Confidence threshold {self.confidence_threshold}")

        dataset.purpose = DatasetPurpose.INFERENCE
        prediction_results, _ = self._infer_model(dataset, inference_parameters)

        self._add_predictions_to_dataset(
            prediction_results, dataset, self.confidence_threshold, process_saliency_maps, explain_predicted_classes
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
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f"not supported export type {export_type}")
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        results = self._export_model(precision, dump_features)
        outputs = results.get("outputs")
        logger.debug(f"results of run_task = {outputs}")
        if outputs is None:
            raise RuntimeError(results.get("msg"))

        bin_file = outputs.get("bin")
        xml_file = outputs.get("xml")
        onnx_file = outputs.get("onnx")

        ir_extra_data = get_det_model_api_configuration(
            self._task_environment.label_schema, self._task_type, self.confidence_threshold
        )
        embed_ir_model_data(xml_file, ir_extra_data)

        if xml_file is None or bin_file is None or onnx_file is None:
            raise RuntimeError("invalid status of exporting. bin and xml or onnx should not be None")
        with open(bin_file, "rb") as f:
            output_model.set_data("openvino.bin", f.read())
        with open(xml_file, "rb") as f:
            output_model.set_data("openvino.xml", f.read())
        with open(onnx_file, "rb") as f:
            output_model.set_data("model.onnx", f.read())

        if self._hyperparams.tiling_parameters.enable_tile_classifier:
            tile_classifier = None
            for partition in outputs.get("partitioned", {}):
                if partition.get("tile_classifier"):
                    tile_classifier = partition.get("tile_classifier")
                    break
            if tile_classifier is None:
                raise RuntimeError("invalid status of exporting. tile_classifier should not be None")
            with open(tile_classifier["bin"], "rb") as f:
                output_model.set_data("tile_classifier.bin", f.read())
            with open(tile_classifier["xml"], "rb") as f:
                output_model.set_data("tile_classifier.xml", f.read())

        output_model.set_data(
            "confidence_threshold",
            np.array([self.confidence_threshold], dtype=np.float32).tobytes(),
        )
        output_model.set_data("config.json", config_to_bytes(self._hyperparams))
        output_model.precision = self._precision
        output_model.optimization_methods = self._optimization_methods
        output_model.has_xai = dump_features
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        logger.info("Exporting completed")

    @abstractmethod
    def _export_model(self, precision: ModelPrecision, dump_features: bool):
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
        logger.info(f"F-measure after evaluation: {metric.f_measure.value}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    def _add_predictions_to_dataset(
        self,
        prediction_results,
        dataset,
        confidence_threshold=0.0,
        process_saliency_maps=False,
        explain_predicted_classes=True,
    ):
        """Loop over dataset again to assign predictions. Convert from MMDetection format to OTX format."""
        for dataset_item, (all_results, feature_vector, saliency_map) in zip(dataset, prediction_results):
            shapes = self._get_shapes(all_results, dataset_item.width, dataset_item.height, confidence_threshold)
            dataset_item.append_annotations(shapes)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if saliency_map is not None:
                labels = self._labels.copy()
                if saliency_map.shape[0] == len(labels) + 1:
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

    def _get_shapes(self, all_results, width, height, confidence_threshold):
        if self._task_type == TaskType.DETECTION:
            shapes = self._det_add_predictions_to_dataset(all_results, width, height, confidence_threshold)
        elif self._task_type in {
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.ROTATED_DETECTION,
        }:
            shapes = self._ins_seg_add_predictions_to_dataset(all_results, width, height, confidence_threshold)
        else:
            raise RuntimeError(f"MPA results assignment not implemented for task: {self._task_type}")
        return shapes

    def _det_add_predictions_to_dataset(self, all_results, width, height, confidence_threshold):
        shapes = []
        for label_idx, detections in enumerate(all_results):
            for i in range(detections.shape[0]):
                probability = float(detections[i, 4])
                coords = detections[i, :4].astype(float).copy()
                coords /= np.array([width, height, width, height], dtype=float)
                coords = np.clip(coords, 0, 1)

                if probability < confidence_threshold:
                    continue

                assigned_label = [ScoredLabel(self._labels[label_idx], probability=probability)]
                if coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0:
                    continue

                shapes.append(
                    Annotation(
                        Rectangle(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                        labels=assigned_label,
                    )
                )
        return shapes

    def _ins_seg_add_predictions_to_dataset(self, all_results, width, height, confidence_threshold):
        shapes = []
        for label_idx, (boxes, masks) in enumerate(zip(*all_results)):
            for mask, probability in zip(masks, boxes[:, 4]):
                if isinstance(mask, dict):
                    mask = mask_util.decode(mask)
                mask = mask.astype(np.uint8)
                probability = float(probability)
                contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchies is None:
                    continue
                for contour, hierarchy in zip(contours, hierarchies[0]):
                    if hierarchy[3] != -1:
                        continue
                    if len(contour) <= 2 or probability < confidence_threshold:
                        continue
                    if self._task_type == TaskType.INSTANCE_SEGMENTATION:
                        points = [Point(x=point[0][0] / width, y=point[0][1] / height) for point in contour]
                    else:
                        box_points = cv2.boxPoints(cv2.minAreaRect(contour))
                        points = [Point(x=point[0] / width, y=point[1] / height) for point in box_points]
                    labels = [ScoredLabel(self._labels[label_idx], probability=probability)]
                    polygon = Polygon(points=points)
                    if cv2.contourArea(contour) > 0 and polygon.get_area() > 1e-12:
                        shapes.append(Annotation(polygon, labels=labels, id=ID(f"{label_idx:08}")))
        return shapes

    def _add_explanations_to_dataset(
        self, detections, explain_results, dataset, process_saliency_maps, explain_predicted_classes
    ):
        """Add saliency map to the dataset."""
        for dataset_item, detection, saliency_map in zip(dataset, detections, explain_results):
            labels = self._labels.copy()
            if saliency_map.shape[0] == len(labels) + 1:
                # Include the background as the last category
                labels.append(LabelEntity("background", Domain.DETECTION))

            shapes = self._get_shapes(detection, dataset_item.width, dataset_item.height, 0.4)
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
            "VERSION": 1,
        }

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        output_model.precision = self._precision
