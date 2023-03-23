"""Train Task of OTX Detection."""

# Copyright (C) 2022 Intel Corporation
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
from typing import Any, Iterable, List, Optional

import numpy as np
import torch
from mmcv.utils import ConfigDict

from otx.algorithms.common.utils.callback import TrainingProgressCallback
from otx.algorithms.common.utils.data import get_dataset
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.utils.config_utils import (
    should_cluster_anchors,
)
from otx.algorithms.detection.utils.data import adaptive_tile_params
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import ids_to_strings
from otx.api.entities.datasets import DatasetEntity
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
from otx.api.entities.model import ModelEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.train_parameters import TrainParameters, default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.training_interface import ITrainingTask
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

from .inference import DetectionInferenceTask

logger = get_logger()


# pylint: disable=too-many-locals, too-many-instance-attributes, too-many-ancestors
class DetectionTrainTask(DetectionInferenceTask, ITrainingTask):
    """Train Task Implementation of OTX Detection."""

    @check_input_parameters_type()
    def save_model(self, output_model: ModelEntity):
        """Save best model weights in DetectionTrainTask."""
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
        if self._recipe_cfg is not None and should_cluster_anchors(self._recipe_cfg):
            modelinfo["anchors"] = {}
            self._update_anchors(modelinfo["anchors"], self._recipe_cfg.model.bbox_head.anchor_generator)

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        output_model.precision = self._precision

    def cancel_training(self):
        """Cancel training function in DetectionTrainTask.

        Sends a cancel training signal to gracefully stop the optimizer. The signal consists of creating a
        '.stop_training' file in the current work_dir. The runner checks for this file periodically.
        The stopping mechanism allows stopping after each iteration, but validation will still be carried out. Stopping
        will therefore take some time.
        """
        logger.info("Cancel training requested.")
        self._should_stop = True
        if self.cancel_interface is not None:
            self.cancel_interface.cancel()
        else:
            logger.info("but training was not started yet. reserved it to cancel")
            self.reserved_cancel = True

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: Optional[TrainParameters] = None,
    ):
        """Train function in DetectionTrainTask."""
        logger.info("train()")
        # Check for stop signal when training has stopped.
        # If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

        # Set OTE LoggerHook & Time Monitor
        if train_parameters:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback
        self._time_monitor = TrainingProgressCallback(update_progress_callback)

        self._data_cfg = self._init_train_data_cfg(dataset)
        self._is_training = True

        if bool(self._hyperparams.tiling_parameters.enable_tiling) and bool(
            self._hyperparams.tiling_parameters.enable_adaptive_params
        ):
            adaptive_tile_params(self._hyperparams.tiling_parameters, dataset)

        results = self._run_task(
            "DetectionTrainer",
            mode="train",
            dataset=dataset,
            parameters=train_parameters,
        )

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
        val_preds, val_map = self._infer_detector(val_dataset, InferenceParameters(is_evaluation=True))

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
        performance.dashboard_metrics.extend(
            DetectionTrainTask._generate_training_metrics(self._learning_curves, val_map)
        )
        logger.info(f"Final model performance: {str(performance)}")
        # save resulting model
        self.save_model(output_model)
        output_model.performance = performance
        # output_model.model_status = ModelStatus.SUCCESS
        logger.info("train done.")

    def _init_train_data_cfg(self, dataset: DatasetEntity):
        logger.info("init data cfg.")
        data_cfg = ConfigDict(data=ConfigDict())

        for cfg_key, subset in zip(
            ["train", "val", "unlabeled"],
            [Subset.TRAINING, Subset.VALIDATION, Subset.UNLABELED],
        ):
            subset = get_dataset(dataset, subset)
            if subset:
                data_cfg.data[cfg_key] = ConfigDict(
                    otx_dataset=subset,
                    labels=self._labels,
                )

        return data_cfg

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
