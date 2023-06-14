"""Task of OTX Video Recognition."""

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
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from mmcv.utils import ConfigDict

from otx.algorithms.action.configs.base import ActionConfig
from otx.algorithms.common.tasks.base_task import TRAIN_TYPE_DIR_PATH, OTXTask
from otx.algorithms.common.utils.callback import (
    InferenceProgressCallback,
    TrainingProgressCallback,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import config_to_bytes, ids_to_strings
from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.explain_parameters import ExplainParameters
from otx.api.entities.inference_parameters import InferenceParameters
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
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.entities.train_parameters import TrainParameters, default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.accuracy import Accuracy
from otx.api.usecases.evaluation.f_measure import FMeasure
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.utils.vis_utils import get_actmap
from otx.cli.utils.multi_gpu import is_multigpu_child_process

logger = get_logger()


class OTXActionTask(OTXTask, ABC):
    """Task class for OTX action."""

    # pylint: disable=too-many-instance-attributes, too-many-locals
    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__(task_environment, output_path)
        self._task_config = ActionConfig
        self._hyperparams: ConfigDict = task_environment.get_hyper_parameters(self._task_config)
        self._train_type = self._hyperparams.algo_backend.train_type
        self._model_dir = os.path.join(
            os.path.abspath(os.path.dirname(self._task_environment.model_template.model_template_path)),
            TRAIN_TYPE_DIR_PATH[self._train_type.name],
        )

        if hasattr(self._hyperparams, "postprocessing") and hasattr(
            self._hyperparams.postprocessing, "confidence_threshold"
        ):
            self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
        else:
            self.confidence_threshold = 0.0

        if task_environment.model is not None:
            self._load_model()

        self.data_pipeline_path = os.path.join(self._model_dir, "data_pipeline.py")

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: Optional[TrainParameters] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        """Train function for OTX action task.

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
        self.seed = seed
        self.deterministic = deterministic

        # Set OTX LoggerHook & Time Monitor
        if train_parameters:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback
        self._time_monitor = TrainingProgressCallback(update_progress_callback)

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
            return
        # update checkpoint to the newly trained model
        self._model_ckpt = model_ckpt

        # get prediction on validation set
        self._is_training = False
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        val_preds, val_performance = self._infer_model(val_dataset, InferenceParameters(is_evaluation=True))

        preds_val_dataset = val_dataset.with_empty_annotations()
        if self._task_type == TaskType.ACTION_CLASSIFICATION:
            self._add_cls_predictions_to_dataset(val_preds, preds_val_dataset)
        elif self._task_type == TaskType.ACTION_DETECTION:
            self._add_det_predictions_to_dataset(val_preds, preds_val_dataset, 0.0)

        result_set = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=val_dataset,
            prediction_dataset=preds_val_dataset,
        )

        metric: Union[Accuracy, FMeasure]

        if self._task_type == TaskType.ACTION_CLASSIFICATION:
            metric = MetricsHelper.compute_accuracy(result_set)
        if self._task_type == TaskType.ACTION_DETECTION:
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
        performance = metric.get_performance()
        performance.dashboard_metrics.extend(self._generate_training_metrics(self._learning_curves, val_performance))
        logger.info(f"Final model performance: {performance}")
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

        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress  # type: ignore

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)
        # If confidence threshold is adaptive then up-to-date value should be stored in the model
        # and should not be changed during inference. Otherwise user-specified value should be taken.
        if not self._hyperparams.postprocessing.result_based_confidence_threshold:
            self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
        logger.info(f"Confidence threshold {self.confidence_threshold}")

        prediction_results, _ = self._infer_model(dataset, inference_parameters)

        if self._task_type == TaskType.ACTION_CLASSIFICATION:
            self._add_cls_predictions_to_dataset(prediction_results, dataset)
        elif self._task_type == TaskType.ACTION_DETECTION:
            self._add_det_predictions_to_dataset(prediction_results, dataset, self.confidence_threshold)
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
        """Export function of OTX Task."""
        if dump_features:
            raise NotImplementedError(
                "Feature dumping is not implemented for the action task."
                "The saliency maps and representation vector outputs will not be dumped in the exported model."
            )

        self._update_model_export_metadata(output_model, export_type, precision, dump_features)
        results = self._export_model(precision, export_type, dump_features)

        outputs = results.get("outputs")
        logger.debug(f"results of run_task = {outputs}")
        if outputs is None:
            raise RuntimeError(results.get("msg"))

        if export_type == ExportType.ONNX:
            onnx_file = outputs.get("onnx")
            with open(onnx_file, "rb") as f:
                output_model.set_data("model.onnx", f.read())
        else:
            bin_file = outputs.get("bin")
            xml_file = outputs.get("xml")

            with open(bin_file, "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            with open(xml_file, "rb") as f:
                output_model.set_data("openvino.xml", f.read())

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
        raise NotImplementedError

    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[ExplainParameters] = None,
    ) -> DatasetEntity:
        """Main explain function of OTX Task."""
        raise NotImplementedError("Video recognition task don't support otx explain yet.")

    def evaluate(
        self,
        output_resultset: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        """Evaluate function of OTX Action Task."""
        logger.info("called evaluate()")
        if evaluation_metric is not None:
            logger.warning(
                f"Requested to use {evaluation_metric} metric, " "but parameter is ignored. Use F-measure instead."
            )
        self._remove_empty_frames(output_resultset.ground_truth_dataset)

        metric: Union[Accuracy, FMeasure]
        if self._task_type == TaskType.ACTION_CLASSIFICATION:
            metric = MetricsHelper.compute_accuracy(output_resultset)
        if self._task_type == TaskType.ACTION_DETECTION:
            metric = MetricsHelper.compute_f_measure(output_resultset)
        performance = metric.get_performance()
        logger.info(f"Final model performance: {str(performance)}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    def _remove_empty_frames(self, dataset: DatasetEntity):
        """Remove empty frame for action detection dataset."""
        remove_indices = []
        for idx, item in enumerate(dataset):
            if item.get_metadata()[0].data.is_empty_frame:
                remove_indices.append(idx)
        dataset.remove_at_indices(remove_indices)

    def _add_cls_predictions_to_dataset(self, prediction_results: Iterable, dataset: DatasetEntity):
        """Loop over dataset again to assign predictions. Convert from MM format to OTX format."""
        prediction_results = list(prediction_results)
        video_info: Dict[str, int] = {}
        for dataset_item in dataset:
            video_id = dataset_item.get_metadata()[0].data.video_id
            if video_id not in video_info:
                video_info[video_id] = len(video_info)
        for dataset_item in dataset:
            video_id = dataset_item.get_metadata()[0].data.video_id
            all_results, feature_vector, saliency_map = prediction_results[video_info[video_id]]
            item_labels = []
            label = ScoredLabel(label=self._labels[all_results.argmax()], probability=all_results.max())
            item_labels.append(label)
            dataset_item.append_labels(item_labels)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if saliency_map is not None:
                saliency_map = get_actmap(saliency_map, (dataset_item.width, dataset_item.height))
                saliency_map_media = ResultMediaEntity(
                    name="Saliency Map",
                    type="saliency_map",
                    annotation_scene=dataset_item.annotation_scene,
                    numpy=saliency_map,
                    roi=dataset_item.roi,
                )
                dataset_item.append_metadata_item(saliency_map_media, model=self._task_environment.model)

    def _add_det_predictions_to_dataset(
        self, prediction_results: Iterable, dataset: DatasetEntity, confidence_threshold: float = 0.05
    ):
        self._remove_empty_frames(dataset)
        for dataset_item, (all_results, feature_vector, saliency_map) in zip(dataset, prediction_results):
            shapes = []
            for label_idx, detections in enumerate(all_results):
                for i in range(detections.shape[0]):
                    probability = float(detections[i, 4])
                    coords = detections[i, :4]

                    if probability < confidence_threshold:
                        continue
                    if coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0:
                        continue

                    assigned_label = [ScoredLabel(self._labels[label_idx], probability=probability)]
                    shapes.append(
                        Annotation(
                            Rectangle(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                            labels=assigned_label,
                        )
                    )
            dataset_item.append_annotations(shapes)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if saliency_map is not None:
                saliency_map = get_actmap(saliency_map, (dataset_item.width, dataset_item.height))
                saliency_map_media = ResultMediaEntity(
                    name="Saliency Map",
                    type="saliency_map",
                    annotation_scene=dataset_item.annotation_scene,
                    numpy=saliency_map,
                    roi=dataset_item.roi,
                )
                dataset_item.append_metadata_item(saliency_map_media, model=self._task_environment.model)

    @staticmethod
    # TODO Implement proper function for action classification
    def _generate_training_metrics(learning_curves, scores, metric_name="mAP") -> Iterable[MetricsGroup[Any, Any]]:
        """Get Training metrics (epochs & scores).

        Parses the mmaction logs to get metrics from the latest training run
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
                metrics=[ScoreMetric(value=scores, name=f"{metric_name}")],
                visualization_info=BarChartInfo("Validation score", visualization_type=VisualizationType.RADIAL_BAR),
            )
        )

        return output

    def save_model(self, output_model: ModelEntity):
        """Save best model weights in ActionTrainTask."""
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
