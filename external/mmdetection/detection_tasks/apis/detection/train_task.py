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
import os
from collections import defaultdict
from glob import glob
from typing import List, Optional

import numpy as np
import torch
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.metrics import (BarChartInfo, BarMetricsGroup, CurveMetric, LineChartInfo, LineMetricsGroup, MetricsGroup,
                                      ScoreMetric, VisualizationType)
from ote_sdk.entities.model import ModelEntity, ModelPrecision
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.train_parameters import TrainParameters, default_progress_callback
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from ote_sdk.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

from mmdet.apis import train_detector
from detection_tasks.apis.detection.config_utils import cluster_anchors, prepare_for_training, set_hyperparams
from detection_tasks.apis.detection.inference_task import OTEDetectionInferenceTask
from detection_tasks.apis.detection.ote_utils import TrainingProgressCallback
from detection_tasks.extension.utils.hooks import OTELoggerHook
from mmdet.datasets import build_dataset
from mmdet.utils.logger import get_root_logger

logger = get_root_logger()


class OTEDetectionTrainingTask(OTEDetectionInferenceTask, ITrainingTask):

    def _generate_training_metrics(self, learning_curves, map) -> Optional[List[MetricsGroup]]:
        """
        Parses the mmdetection logs to get metrics from the latest training run

        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        # Learning curves.
        for key, curve in learning_curves.items():
            n, m = len(curve.x), len(curve.y)
            if n != m:
                logger.warning(f"Learning curve {key} has inconsistent number of coordinates ({n} vs {m}.")
                n = min(n, m)
                curve.x = curve.x[:n]
                curve.y = curve.y[:n]
            metric_curve = CurveMetric(
                xs=np.nan_to_num(curve.x).tolist(),
                ys=np.nan_to_num(curve.y).tolist(),
                name=key)
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(LineMetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        # Final mAP value on the validation set.
        output.append(
            BarMetricsGroup(
                metrics=[ScoreMetric(value=map, name="mAP")],
                visualization_info=BarChartInfo("Validation score", visualization_type=VisualizationType.RADIAL_BAR)
            )
        )

        return output


    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def train(self, dataset: DatasetEntity, output_model: ModelEntity, train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        logger.info('Training the model')
        set_hyperparams(self._config, self._hyperparams)

        train_dataset = dataset.get_subset(Subset.TRAINING)
        val_dataset = dataset.get_subset(Subset.VALIDATION)

        # Do clustering for SSD model
        if hasattr(self._config.model, 'bbox_head') and hasattr(self._config.model.bbox_head, 'anchor_generator'):
            if getattr(self._config.model.bbox_head.anchor_generator, 'reclustering_anchors', False):
                self._config, self._model = cluster_anchors(self._config, train_dataset, self._model)

        config = self._config

        # Create a copy of the network.
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
        update_progress_callback = default_progress_callback
        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
        time_monitor = TrainingProgressCallback(update_progress_callback)
        learning_curves = defaultdict(OTELoggerHook.Curve)
        training_config = prepare_for_training(config, train_dataset, val_dataset, time_monitor, learning_curves)
        self._training_work_dir = training_config.work_dir
        mm_train_dataset = build_dataset(training_config.data.train)
        self._is_training = True
        self._model.train()
        logger.info('Start training')
        train_detector(model=self._model, dataset=mm_train_dataset, cfg=training_config, validate=True)
        logger.info('Training completed')

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        # model should be returned. Old train model is restored.
        if self._should_stop:
            logger.info('Training cancelled.')
            self._model = old_model
            self._should_stop = False
            self._is_training = False
            return

        # Load best weights.
        checkpoint_file_path = glob(os.path.join(training_config.work_dir, 'best*pth'))
        if len(checkpoint_file_path) == 0:
            checkpoint_file_path = os.path.join(training_config.work_dir, 'latest.pth')
        elif len(checkpoint_file_path) > 1:
            logger.warning(f'Multiple candidates for the best checkpoint found: {checkpoint_file_path}')
            checkpoint_file_path = checkpoint_file_path[0]
        else:
            checkpoint_file_path = checkpoint_file_path[0]
        logger.info(f'Use {checkpoint_file_path} for final model weights.')
        checkpoint = torch.load(checkpoint_file_path)
        self._model.load_state_dict(checkpoint['state_dict'])

        # Get predictions on the validation set.
        val_preds, val_map = self._infer_detector(
            self._model,
            config,
            val_dataset,
            metric_name=config.evaluation.metric,
            dump_features=False,
            eval=True
        )
        preds_val_dataset = val_dataset.with_empty_annotations()
        self._add_predictions_to_dataset(val_preds, preds_val_dataset, 0.0)
        resultset = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=val_dataset,
            prediction_dataset=preds_val_dataset,
        )

        # Adjust confidence threshold.
        adaptive_threshold = self._hyperparams.postprocessing.result_based_confidence_threshold
        if adaptive_threshold:
            logger.info('Adjusting the confidence threshold')
            metric = MetricsHelper.compute_f_measure(resultset, vary_confidence_threshold=True)
            best_confidence_threshold = metric.best_confidence_threshold.value
            if best_confidence_threshold is None:
                raise ValueError(f"Cannot compute metrics: Invalid confidence threshold!")
            logger.info(f"Setting confidence threshold to {best_confidence_threshold} based on results")
            self.confidence_threshold = best_confidence_threshold
        else:
            metric = MetricsHelper.compute_f_measure(resultset, vary_confidence_threshold=False)

        # Compose performance statistics.
        # TODO[EUGENE]: ADD MAE CURVE FOR TaskType.COUNTING
        performance = metric.get_performance()
        performance.dashboard_metrics.extend(self._generate_training_metrics(learning_curves, val_map))
        logger.info(f'Final model performance: {str(performance)}')

        # Save resulting model.
        self.save_model(output_model)
        output_model.performance = performance

        self._is_training = False
        logger.info('Training the model [done]')


    @check_input_parameters_type()
    def save_model(self, output_model: ModelEntity):
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))

        modelinfo = {'model': self._model.state_dict(),
                     'config': hyperparams_str,
                     'confidence_threshold': self.confidence_threshold,
                     'VERSION': 1}

        if hasattr(self._config.model, 'bbox_head') and hasattr(self._config.model.bbox_head, 'anchor_generator'):
            if getattr(self._config.model.bbox_head.anchor_generator, 'reclustering_anchors', False):
                generator = self._model.bbox_head.anchor_generator
                modelinfo['anchors'] = {'heights': generator.heights, 'widths': generator.widths}

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
        output_model.precision = self._precision


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
