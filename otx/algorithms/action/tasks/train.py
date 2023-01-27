"""Train Task of OTX Action Task."""

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
import os
from glob import glob
from typing import Any, Iterable, List, Optional

import numpy as np
import torch
from mmaction.apis import train_model
from mmaction.datasets import build_dataset
from mmaction.utils import get_root_logger

from otx.algorithms.action.adapters.mmaction.utils import prepare_for_training
from otx.algorithms.common.utils import TrainingProgressCallback
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
from otx.api.entities.model_template import TaskType
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.train_parameters import TrainParameters, default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.tasks.interfaces.training_interface import ITrainingTask

from .inference import ActionInferenceTask

logger = get_root_logger()


# pylint: disable=too-many-locals, too-many-instance-attributes
class ActionTrainTask(ActionInferenceTask, ITrainingTask):
    """Train Task Implementation of OTX Action Task."""

    def save_model(self, output_model: ModelEntity):
        """Save best model weights in ActionTrainTask."""
        logger.info("called save_model")
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {
            "model": model_ckpt["state_dict"],
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

    def cancel_training(self):
        """Cancel training function in ActionTrainTask.

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

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: Optional[TrainParameters] = None,
    ):
        """Train function in ActionTrainTask."""
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

        self._is_training = True
        self._init_task()

        if self._recipe_cfg is None:
            raise Exception("Recipe config is not initialized properly")

        results = self._train_model(dataset)

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

        self._get_output_model(results)
        performance = self._get_final_eval_results(dataset, output_model)

        # save resulting model
        self.save_model(output_model)
        output_model.performance = performance
        self._is_training = False
        logger.info("train done.")

    def _train_model(self, dataset: DatasetEntity):
        if self._recipe_cfg is None:
            raise Exception("Recipe config does not initialize properly!")
        train_dataset = dataset.get_subset(Subset.TRAINING)
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        training_config = prepare_for_training(
            self._recipe_cfg, train_dataset, val_dataset, self._time_monitor, self._learning_curves
        )
        mm_train_dataset = build_dataset(training_config.data.train)
        logger.info("Start training")
        self._model.train()
        # FIXME runner is built inside of train_model funciton, it is hard to change runner's type
        train_model(model=self._model, dataset=mm_train_dataset, cfg=training_config, validate=True)
        checkpoint_file_path = glob(os.path.join(self._recipe_cfg.work_dir, "best*pth"))
        if len(checkpoint_file_path) == 0:
            checkpoint_file_path = os.path.join(self._recipe_cfg.work_dir, "latest.pth")
        elif len(checkpoint_file_path) > 1:
            logger.warning(f"Multiple candidates for the best checkpoint found: {checkpoint_file_path}")
            checkpoint_file_path = checkpoint_file_path[0]
        else:
            checkpoint_file_path = checkpoint_file_path[0]
        logger.info(f"Use {checkpoint_file_path} for final model weights")

        return {"final_ckpt": checkpoint_file_path}

    def _get_output_model(self, results):
        model_ckpt = results.get("final_ckpt")
        if model_ckpt is None:
            logger.error("cannot find final checkpoint from the results.")
            return
        # update checkpoint to the newly trained model
        self._model_ckpt = model_ckpt
        self._model.load_state_dict(torch.load(self._model_ckpt)["state_dict"])

    def _get_final_eval_results(self, dataset, output_model):
        logger.info("Final Evaluation")
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        val_preds, val_map = self._infer_model(val_dataset, InferenceParameters(is_evaluation=True))

        preds_val_dataset = val_dataset.with_empty_annotations()
        # TODO Load _add_predictions_to_dataset function from self._task_type
        if self._task_type == TaskType.ACTION_CLASSIFICATION:
            self._add_predictions_to_dataset(val_preds, preds_val_dataset)
        elif self._task_type == TaskType.ACTION_DETECTION:
            self._add_det_predictions_to_dataset(val_preds, preds_val_dataset)

        result_set = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=val_dataset,
            prediction_dataset=preds_val_dataset,
        )

        metric = self._get_metric(result_set)

        # compose performance statistics
        performance = metric.get_performance()
        metric_name = self._recipe_cfg.evaluation.final_metric
        performance.dashboard_metrics.extend(
            ActionTrainTask._generate_training_metrics(self._learning_curves, val_map, metric_name)
        )
        logger.info(f"Final model performance: {str(performance)}")
        return performance

    @staticmethod
    # TODO Implement proper function for action classification
    def _generate_training_metrics(learning_curves, scores, metric_name) -> Iterable[MetricsGroup[Any, Any]]:
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
