# Copyright (C) 2021 Intel Corporation
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

import copy
import io
import logging
import os
from collections import defaultdict
from typing import List, Optional

import torch
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.metrics import (CurveMetric, InfoMetric, LineChartInfo, MetricsGroup, Performance, ScoreMetric,
                                      VisualizationInfo, VisualizationType)
from ote_sdk.entities.model import ModelEntity, ModelPrecision
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.entities.train_parameters import default_progress_callback as default_train_progress_callback
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from ote_sdk.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

from mmseg.apis import train_segmentor
from segmentation_tasks.apis.segmentation.config_utils import prepare_for_training, set_hyperparams
from segmentation_tasks.apis.segmentation.ote_utils import TrainingProgressCallback
from segmentation_tasks.extension.utils.hooks import OTELoggerHook
from mmseg.datasets import build_dataset
from segmentation_tasks.apis.segmentation import OTESegmentationInferenceTask

logger = logging.getLogger(__name__)


class OTESegmentationTrainingTask(OTESegmentationInferenceTask, ITrainingTask):

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def train(self, dataset: DatasetEntity,
              output_model: ModelEntity,
              train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        set_hyperparams(self._config, self._hyperparams)

        train_dataset = dataset.get_subset(Subset.TRAINING)
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        config = self._config

        # Create new model if training from scratch.
        old_model = copy.deepcopy(self._model)

        # Check for stop signal between pre-eval and training. If training is cancelled at this point,
        # old_model should be restored.
        if self._should_stop:
            logger.info('Training cancelled.')
            self._model = old_model
            self._should_stop = False
            self._is_training = False
            self._training_work_dir = None
            return

        # Run training.
        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_train_progress_callback
        time_monitor = TrainingProgressCallback(update_progress_callback)
        learning_curves = defaultdict(OTELoggerHook.Curve)
        training_config = prepare_for_training(config, train_dataset, val_dataset, time_monitor, learning_curves)
        self._training_work_dir = training_config.work_dir
        mm_train_dataset = build_dataset(training_config.data.train)
        self._is_training = True
        self._model.train()

        train_segmentor(model=self._model, dataset=mm_train_dataset, cfg=training_config, validate=True)

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        # model should be returned. Old train model is restored.
        if self._should_stop:
            logger.info('Training cancelled.')
            self._model = old_model
            self._should_stop = False
            self._is_training = False
            return

        # Load the best weights and check if model has improved.
        best_checkpoint_path = self._find_best_checkpoint(training_config.work_dir, config.evaluation.metric)
        best_checkpoint = torch.load(best_checkpoint_path)
        self._model.load_state_dict(best_checkpoint['state_dict'])

        # Add loss curves
        training_metrics, best_score = self._generate_training_metrics_group(learning_curves)
        performance = Performance(score=ScoreMetric(value=best_score, name="mDice"),
                                  dashboard_metrics=training_metrics)

        self.save_model(output_model)
        output_model.performance = performance

        self._is_training = False

    @staticmethod
    def _find_best_checkpoint(work_dir, metric):
        all_files = [f for f in os.listdir(work_dir) if os.path.isfile(os.path.join(work_dir, f))]

        name_prefix = f'best_{metric}_'
        candidates = [f for f in all_files if f.startswith(name_prefix) and f.endswith('.pth')]

        if len(candidates) == 0:
            out_name = 'latest.pth'
        else:
            assert len(candidates) == 1
            out_name = candidates[0]

        return os.path.join(work_dir, out_name)

    @check_input_parameters_type()
    def save_model(self, output_model: ModelEntity):
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_info = {'model': self._model.state_dict(), 'config': hyperparams_str, 'labels': labels, 'VERSION': 1}

        buffer = io.BytesIO()
        torch.save(model_info, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
        output_model.precision = [ModelPrecision.FP32]


    def cancel_training(self):
        """
        Sends a cancel training signal to gracefully stop the optimizer. The signal consists of creating a
        '.stop_training' file in the current work_dir. The runner checks for this file periodically.
        The stopping mechanism allows stopping after each iteration, but validation will still be carried out. Stopping
        will therefore take some time.
        """
        logger.info("Cancel training requested.")
        self._should_stop = True
        stop_training_filepath = os.path.join(self._training_work_dir, '.stop_training')
        open(stop_training_filepath, 'a').close()

    def _generate_training_metrics_group(self, learning_curves) -> Optional[List[MetricsGroup]]:
        """
        Parses the mmsegmentation logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []
        metric_key = 'val/mDice'

        # Model architecture
        architecture = InfoMetric(name='Model architecture', value=self._model_name)
        visualization_info_architecture = VisualizationInfo(name="Model architecture",
                                                            visualisation_type=VisualizationType.TEXT)
        output.append(MetricsGroup(metrics=[architecture],
                                   visualization_info=visualization_info_architecture))

        # Learning curves
        for key, curve in learning_curves.items():
            if key == metric_key:
                best_score = max(curve.y)
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(MetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        return output, best_score
