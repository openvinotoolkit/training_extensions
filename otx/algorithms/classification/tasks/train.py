"""Train Task of OTX Classification."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name

import io
from typing import List, Optional

import torch
from mmcv.utils import ConfigDict

from otx.algorithms.classification.configs import ClassificationConfig
from otx.algorithms.common.utils.callback import TrainingProgressCallback
from otx.algorithms.common.utils.data import get_dataset
from otx.algorithms.common.utils.logger import get_logger
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import ids_to_strings
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.metrics import (
    CurveMetric,
    LineChartInfo,
    LineMetricsGroup,
    MetricsGroup,
    Performance,
    ScoreMetric,
)
from otx.api.entities.model import ModelEntity  # ModelStatus
from otx.api.entities.subset import Subset
from otx.api.entities.train_parameters import TrainParameters
from otx.api.entities.train_parameters import (
    default_progress_callback as train_default_progress_callback,
)
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

from .inference import ClassificationInferenceTask

logger = get_logger()

TASK_CONFIG = ClassificationConfig


# pylint: disable= too-many-ancestors
class ClassificationTrainTask(ClassificationInferenceTask):
    """Train Task Implementation of OTX Classification."""

    @check_input_parameters_type()
    def save_model(self, output_model: ModelEntity):
        """Save best model weights in ClassificationTrainTask."""
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

    def cancel_training(self):
        """Cancel training function in ClassificationTrainTask.

        Sends a cancel training signal to gracefully stop the optimizer.
        The signal consists of creating a '.stop_training' file in the current work_dir.
        The runner checks for this file periodically.
        The stopping mechanism allows stopping after each iteration, but validation will still be carried out.
        Stopping will therefore take some time.
        """
        self._should_stop = True
        logger.info("Cancel training requested.")
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
        """Train function in ClassificationTrainTask."""
        logger.info("train()")
        # Check for stop signal between pre-eval and training.
        # If training is cancelled at this point,
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

        # Set OTX LoggerHook & Time Monitor
        update_progress_callback = train_default_progress_callback
        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress  # type: ignore
        self._time_monitor = TrainingProgressCallback(update_progress_callback)

        stage_module = "ClsTrainer"
        self._data_cfg = self._init_train_data_cfg(dataset)
        self._is_training = True
        results = self._run_task(stage_module, mode="train", dataset=dataset, parameters=train_parameters)

        # Check for stop signal between pre-eval and training.
        # If training is cancelled at this point,
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

        # compose performance statistics
        training_metrics, final_acc = self._generate_training_metrics_group(self._learning_curves)
        # save resulting model
        self.save_model(output_model)
        performance = Performance(
            score=ScoreMetric(value=final_acc, name="accuracy"),
            dashboard_metrics=training_metrics,
        )
        logger.info(f"Final model performance: {str(performance)}")
        output_model.performance = performance
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
        """Parses the classification logs to get metrics from the latest training run.

        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        if self._multilabel:
            metric_key = "val/accuracy-mlc"
        elif self._hierarchical:
            metric_key = "val/MHAcc"
        else:
            metric_key = "val/accuracy_top-1"

        # Learning curves
        best_acc = -1
        if learning_curves is None:
            return output

        for key, curve in learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            if key == metric_key:
                best_acc = max(curve.y)
            visualization_info = LineChartInfo(name=key, x_axis_label="Timestamp", y_axis_label=key)
            output.append(LineMetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        return output, best_acc
