"""Task of OTX Segmentation."""

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
from typing import Any, Iterable, List, Optional

import numpy as np
import torch
from mmcv.utils import ConfigDict

from otx.algorithms.common.configs.training_base import TrainType
from otx.algorithms.common.tasks.base_task import TRAIN_TYPE_DIR_PATH, OTXTask
from otx.algorithms.common.utils.callback import (
    InferenceProgressCallback,
    TrainingProgressCallback,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.openvino.model_wrappers.blur import (
    get_activation_map,
)
from otx.algorithms.segmentation.configs.base import SegmentationConfig
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import ids_to_strings
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
from otx.api.entities.model import ModelEntity, ModelPrecision
from otx.api.entities.model_template import TaskType
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.entities.train_parameters import TrainParameters, default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
    create_hard_prediction_from_soft_prediction,
)

logger = get_logger()


class OTXSegmentationTask(OTXTask, ABC):
    """Task class for OTX segmentation."""

    # pylint: disable=too-many-instance-attributes, too-many-locals
    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__(task_environment, output_path)
        self._task_config = SegmentationConfig
        self._hyperparams: ConfigDict = task_environment.get_hyper_parameters(self._task_config)
        self._train_type = self._hyperparams.algo_backend.train_type
        self.metric = "mDice"
        self._label_dictionary = dict(enumerate(sorted(self._labels), 1))

        self._model_dir = os.path.join(
            os.path.abspath(os.path.dirname(self._task_environment.model_template.model_template_path)),
            TRAIN_TYPE_DIR_PATH[self._train_type.name],
        )

        if task_environment.model is not None:
            self._load_model()

        self.data_pipeline_path = os.path.join(self._model_dir, "data_pipeline.py")

    def _load_model_ckpt(self, model: Optional[ModelEntity]):
        if model and "weights.pth" in model.model_adapters:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))
            return model_data
        return None

    def train(
        self, dataset: DatasetEntity, output_model: ModelEntity, train_parameters: Optional[TrainParameters] = None
    ):
        """Train function for OTX segmentation task.

        Actual training is processed by _train_model fucntion
        """
        logger.info("train()")
        # Check for stop signal when training has stopped.
        # If should_stop is true, training was cancelled and no new
        if self._should_stop:  # type: ignore
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

        if self._train_type == TrainType.Selfsupervised:
            self.save_model(output_model)
            logger.info("train done.")
            return

        val_dataset = dataset.get_subset(Subset.VALIDATION)
        pred_dataset = val_dataset.with_empty_annotations()
        predictions = self._infer_model(val_dataset, InferenceParameters(is_evaluation=True))
        prediction_results = zip(predictions["eval_predictions"], predictions["feature_vectors"])

        self._add_predictions_to_dataset(prediction_results, pred_dataset, dump_soft_prediction=False)

        output_resultset = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=val_dataset,
            prediction_dataset=pred_dataset,
        )

        logger.info("Computing mDice")
        metrics = MetricsHelper.compute_dice_averaged_over_pixels(output_resultset)
        logger.info(f"mDice after evaluation: {metrics.overall_dice.value}")
        output_resultset.performance = metrics.get_performance()

        # save resulting model
        self.save_model(output_model)
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

        predictions = self._infer_model(dataset, InferenceParameters(is_evaluation=True))
        prediction_results = zip(predictions["eval_predictions"], predictions["feature_vectors"])
        self._add_predictions_to_dataset(prediction_results, dataset, dump_soft_prediction=False)

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

    @abstractmethod
    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = True,
    ):
        """Export function of OTX Task."""
        raise NotImplementedError

    @abstractmethod
    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[ExplainParameters] = None,
    ) -> DatasetEntity:
        """Main explain function of OTX Task."""
        raise NotImplementedError

    def evaluate(
        self,
        output_resultset: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        """Evaluate function of OTX Segmentation Task."""
        logger.info("called evaluate()")
        if evaluation_metric is not None:
            logger.warning(
                f"Requested to use {evaluation_metric} metric, " "but parameter is ignored. Use mDice instead."
            )
        metric = MetricsHelper.compute_dice_averaged_over_pixels(output_resultset)
        logger.info(f"mDice after evaluation: {metric.overall_dice.value}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    def _add_predictions_to_dataset(self, prediction_results, dataset, dump_soft_prediction):
        """Loop over dataset again to assign predictions. Convert from MMSegmentation format to OTX format."""

        for dataset_item, (prediction, feature_vector) in zip(dataset, prediction_results):
            soft_prediction = np.transpose(prediction[0], axes=(1, 2, 0))
            hard_prediction = create_hard_prediction_from_soft_prediction(
                soft_prediction=soft_prediction,
                soft_threshold=self._hyperparams.postprocessing.soft_threshold,
                blur_strength=self._hyperparams.postprocessing.blur_strength,
            )
            annotations = create_annotation_from_segmentation_map(
                hard_prediction=hard_prediction,
                soft_prediction=soft_prediction,
                label_map=self._label_dictionary,
            )
            dataset_item.append_annotations(annotations=annotations)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if dump_soft_prediction:
                for label_index, label in self._label_dictionary.items():
                    if label_index == 0:
                        continue
                    current_label_soft_prediction = soft_prediction[:, :, label_index]
                    class_act_map = get_activation_map(current_label_soft_prediction)
                    result_media = ResultMediaEntity(
                        name=label.name,
                        type="soft_prediction",
                        label=label,
                        annotation_scene=dataset_item.annotation_scene,
                        roi=dataset_item.roi,
                        numpy=class_act_map,
                    )
                    dataset_item.append_metadata_item(result_media, model=self._task_environment.model)

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

    @staticmethod
    def _generate_training_metrics(learning_curves, scores) -> Iterable[MetricsGroup[Any, Any]]:
        """Get Training metrics (epochs & scores).

        Parses the mmsegmentation logs to get metrics from the latest training run
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
        """Save best model weights in SegmentationTrainTask."""
        logger.info("called save_model")
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {
            "model": model_ckpt,
            "config": hyperparams_str,
            "labels": labels,
            "VERSION": 1,
        }

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        output_model.precision = self._precision
