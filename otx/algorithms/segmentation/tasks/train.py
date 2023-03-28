"""Train Task of OTX Segmentation."""

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
from typing import List, Optional

import torch
from mmcv.utils import ConfigDict

from otx.algorithms.common.utils.callback import TrainingProgressCallback
from otx.algorithms.common.utils.data import get_dataset
from otx.algorithms.common.utils.logger import get_logger
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import ids_to_strings
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.metrics import (
    CurveMetric,
    InfoMetric,
    LineChartInfo,
    MetricsGroup,
    Performance,
    ScoreMetric,
    VisualizationInfo,
    VisualizationType,
)
from otx.api.entities.model import ModelEntity
from otx.api.entities.subset import Subset
from otx.api.entities.train_parameters import TrainParameters, default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.tasks.interfaces.training_interface import ITrainingTask
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

from .inference import SegmentationInferenceTask

logger = get_logger()


# pylint: disable=too-many-locals, too-many-instance-attributes, too-many-ancestors
class SegmentationTrainTask(SegmentationInferenceTask, ITrainingTask):
    """Train Task Implementation of OTX Segmentation."""

    @check_input_parameters_type()
    def save_model(self, output_model: ModelEntity):
        """Save best model weights in SegmentationTrainTask."""
        logger.info(f"called save_model: {self._model_ckpt}")
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

    def cancel_training(self):
        """Cancel training function in SegmentationTrainTask.

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
        """Train function in SegmentationTrainTask."""
        logger.info("train()")
        # Check for stop signal between pre-eval and training.
        # If training is cancelled at this point,
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

        # Set OTX LoggerHook & Time Monitor
        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback
        self._time_monitor = TrainingProgressCallback(update_progress_callback)

        self._data_cfg = self._init_train_data_cfg(dataset)
        self._is_training = True
        results = self._run_task("SegTrainer", mode="train", dataset=dataset, parameters=train_parameters)

        # Check for stop signal when training has stopped.
        # If should_stop is true, training was cancelled and no new
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

        # Get training metrics group from learning curves
        training_metrics, best_score = self._generate_training_metrics_group(self._learning_curves)
        performance = Performance(
            score=ScoreMetric(value=best_score, name=self.metric),
            dashboard_metrics=training_metrics,
        )

        logger.info(f"Final model performance: {str(performance)}")
        # save resulting model
        self.save_model(output_model)
        output_model.performance = performance
        # output_model.model_status = ModelStatus.SUCCESS
        self._is_training = False
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

    def _generate_training_metrics_group(self, learning_curves):
        """Get Training metrics (epochs & scores).

        Parses the mmsegmentation logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []
        # Model architecture
        architecture = InfoMetric(name="Model architecture", value=self._model_name)
        visualization_info_architecture = VisualizationInfo(
            name="Model architecture", visualisation_type=VisualizationType.TEXT
        )
        output.append(
            MetricsGroup(
                metrics=[architecture],
                visualization_info=visualization_info_architecture,
            )
        )
        # Learning curves
        best_score = -1
        for key, curve in learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            if key == f"val/{self.metric}":
                best_score = max(curve.y)
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(MetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))
        return output, best_score
