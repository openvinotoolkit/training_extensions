# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import io
import os
from collections import defaultdict
from typing import List, Optional

import torch
from mpa import MPAConstants

from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings

from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters, default_progress_callback
from ote_sdk.entities.train_parameters import default_progress_callback as train_default_progress_callback
from ote_sdk.entities.model import ModelEntity, ModelPrecision  # ModelStatus
from ote_sdk.entities.resultset import ResultSetEntity
from mmcv.utils import ConfigDict

from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.entities.metrics import (CurveMetric, LineChartInfo,
                                      LineMetricsGroup, MetricsGroup,
                                      Performance, ScoreMetric)
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from ote_sdk.entities.model import (ModelFormat, ModelOptimizationType)
from ote_sdk.serialization.label_mapper import label_schema_to_bytes

from ote_sdk.entities.scored_label import ScoredLabel
from detection_tasks.apis.detection.ote_utils import TrainingProgressCallback
from detection_tasks.extension.utils.hooks import OTELoggerHook
from mpa_tasks.apis import BaseTask, TrainType
from mpa_tasks.apis.classification import ClassificationConfig
from mpa.utils.config_utils import MPAConfig
from mpa.utils.logger import get_logger
from ote_sdk.entities.label import Domain

logger = get_logger()

TASK_CONFIG = ClassificationConfig


class ClassificationInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IUnload):
    def __init__(self, task_environment: TaskEnvironment):
        self._should_stop = False
        super().__init__(TASK_CONFIG, task_environment)

    def infer(self,
              dataset: DatasetEntity,
              inference_parameters: Optional[InferenceParameters] = None
              ) -> DatasetEntity:
        logger.info('called infer()')
        stage_module = 'ClsInferrer'
        self._data_cfg = self._init_test_data_cfg(dataset)
        dataset = dataset.with_empty_annotations()

        results = self._run_task(stage_module, mode='train', dataset=dataset)
        logger.debug(f'result of run_task {stage_module} module = {results}')
        predictions = results['outputs']

        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
        dataset_size = len(dataset)
        for i, (dataset_item, prediction_item) in enumerate(zip(dataset, predictions)):
            label_idx = prediction_item.argmax()
            probability = prediction_item[label_idx]
            dataset_item.append_labels([ScoredLabel(self._labels[label_idx], probability=probability)])
            update_progress_callback(int(i / dataset_size * 100))
        return dataset

    def evaluate(self,
                 output_result_set: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        logger.info('called evaluate()')
        metric = MetricsHelper.compute_accuracy(output_result_set)  # But this line shows proper accuracy
        logger.info(f"Accuracy after evaluation: {metric.accuracy.value}")
        output_result_set.performance = metric.get_performance()
        logger.info('Evaluation completed')

    def unload(self):
        logger.info('called unload()')
        self.finalize()

    def export(self,
               export_type: ExportType,
               output_model: ModelEntity):
        logger.info('Exporting the model')
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f'not supported export type {export_type}')
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        stage_module = 'ClsExporter'
        results = self._run_task(stage_module, mode='train')
        logger.debug(f'results of run_task = {results}')
        results = results.get('outputs')
        logger.debug(f'results of run_task = {results}')
        if results is None:
            logger.error(f"error while exporting model {results.get('msg')}")
        else:
            bin_file = results.get('bin')
            xml_file = results.get('xml')
            if xml_file is None or bin_file is None:
                raise RuntimeError('invalid status of exporting. bin and xml should not be None')
            with open(bin_file, "rb") as f:
                output_model.set_data('openvino.bin', f.read())
            with open(xml_file, "rb") as f:
                output_model.set_data('openvino.xml', f.read())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
        logger.info('Exporting completed')

    def _init_recipe_hparam(self) -> dict:
        return ConfigDict(
            optimizer=ConfigDict(lr=self._hyperparams.learning_parameters.learning_rate),
            lr_config=ConfigDict(warmup_iters=int(self._hyperparams.learning_parameters.learning_rate_warmup_iters)),
            data=ConfigDict(
                samples_per_gpu=int(self._hyperparams.learning_parameters.batch_size),
                workers_per_gpu=int(self._hyperparams.learning_parameters.num_workers),
            ),
            runner=ConfigDict(max_epochs=int(self._hyperparams.learning_parameters.num_iters)),
        )

    def _init_recipe(self):
        logger.info('called _init_recipe()')

        recipe_root = os.path.join(MPAConstants.RECIPES_PATH, 'stages/classification')
        train_type = self._hyperparams.algo_backend.train_type
        logger.info(f'train type = {train_type}')

        recipe = os.path.join(recipe_root, 'semisl.yaml')
        if train_type == TrainType.SemiSupervised:
            recipe = os.path.join(recipe_root, 'semisl.yaml')
        elif train_type == TrainType.SelfSupervised:
            raise NotImplementedError(f'train type {train_type} is not implemented yet.')
        elif train_type == TrainType.Incremental:
            recipe = os.path.join(recipe_root, 'class_incr.yaml')
        else:
            raise NotImplementedError(f'train type {train_type} is not implemented yet.')

        self._recipe_cfg = MPAConfig.fromfile(recipe)
        self._patch_datasets(self._recipe_cfg)  # for OTE compatibility
        logger.info(f'initialized recipe = {recipe}')

    def _init_model_cfg(self):
        base_dir = os.path.abspath(os.path.dirname(self.template_file_path))
        return MPAConfig.fromfile(os.path.join(base_dir, 'model.py'))

    def _init_test_data_cfg(self, dataset: DatasetEntity):
        data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    ote_dataset=dataset,
                    labels=self._labels,
                ),
                test=ConfigDict(
                    ote_dataset=dataset,
                    labels=self._labels,
                )
            )
        )
        return data_cfg

    @staticmethod
    def _patch_datasets(config: MPAConfig, domain=Domain.CLASSIFICATION):
        def patch_color_conversion(pipeline):
            # Default data format for OTE is RGB, while mmdet uses BGR, so negate the color conversion flag.
            for pipeline_step in pipeline:
                if pipeline_step.type == 'Normalize':
                    to_rgb = False
                    if 'to_rgb' in pipeline_step:
                        to_rgb = pipeline_step.to_rgb
                    to_rgb = not bool(to_rgb)
                    pipeline_step.to_rgb = to_rgb
                elif pipeline_step.type == 'MultiScaleFlipAug':
                    patch_color_conversion(pipeline_step.transforms)

        assert 'data' in config
        for subset in ('train', 'val', 'test'):
            cfg = config.data.get(subset, None)
            if not cfg:
                continue
            if cfg.type == 'RepeatDataset':
                cfg = cfg.dataset
            cfg.type = 'MPAClsDataset'
            cfg.domain = domain
            cfg.ote_dataset = None
            cfg.labels = None
            patch_color_conversion(cfg.pipeline)


class ClassificationTrainTask(ClassificationInferenceTask):
    def save_model(self, output_model: ModelEntity):
        logger.info('called save_model')
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {'model': model_ckpt['state_dict'], 'config': hyperparams_str, 'labels': labels, 'VERSION': 1}

        torch.save(modelinfo, buffer)
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
        self._should_stop = True
        logger.info("Cancel training requested.")
        if self.cancel_interface is not None:
            self.cancel_interface.cancel()
        else:
            logger.info('but training was not started yet. reserved it to cancel')
            self.reserved_cancel = True

    def train(self,
              dataset: DatasetEntity,
              output_model: ModelEntity,
              train_parameters: Optional[TrainParameters] = None):
        logger.info('train()')
        # Check for stop signal between pre-eval and training. 
        # If training is cancelled at this point,
        if self._should_stop:
            logger.info('Training cancelled.')
            self._should_stop = False
            self._is_training = False
            return

        # Set OTE LoggerHook & Time Monitor
        update_progress_callback = train_default_progress_callback
        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
        self._time_monitor = TrainingProgressCallback(update_progress_callback)
        self._learning_curves = defaultdict(OTELoggerHook.Curve)

        stage_module = 'ClsTrainer'
        self._data_cfg = self._init_train_data_cfg(dataset)
        self._is_training = True
        results = self._run_task(stage_module, mode='train', dataset=dataset, parameters=train_parameters)

        # Check for stop signal between pre-eval and training. 
        # If training is cancelled at this point,
        if self._should_stop:
            logger.info('Training cancelled.')
            self._should_stop = False
            self._is_training = False
            return

        # get output model
        model_ckpt = results.get('final_ckpt')
        if model_ckpt is None:
            logger.error('cannot find final checkpoint from the results.')
            return
        else:
            # update checkpoint to the newly trained model
            self._model_ckpt = model_ckpt

        # compose performance statistics
        training_metrics, final_acc = self._generate_training_metrics_group(self._learning_curves)
        # save resulting model
        self.save_model(output_model)
        performance = Performance(score=ScoreMetric(value=final_acc, name="accuracy"),
                                  dashboard_metrics=training_metrics)
        logger.info(f'Final model performance: {str(performance)}')
        output_model.performance = performance
        self._is_training = False
        logger.info('train done.')

    def _init_train_data_cfg(self, dataset: DatasetEntity):
        logger.info('init data cfg.')
        data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    ote_dataset=dataset.get_subset(Subset.TRAINING),
                    labels=self._labels,
                    label_names=list(label.name for label in self._labels),
                ),
                val=ConfigDict(
                    ote_dataset=dataset.get_subset(Subset.VALIDATION),
                    labels=self._labels,
                ),
            )
        )

        for label in self._labels:
            label.hotkey = 'a'
        return data_cfg

    def _generate_training_metrics_group(self, learning_curves) -> Optional[List[MetricsGroup]]:
            """
            Parses the classification logs to get metrics from the latest training run
            :return output List[MetricsGroup]
            """
            output: List[MetricsGroup] = []
            metric_key = 'val/accuracy_top-1'

            # Learning curves
            best_acc = -1
            if learning_curves is None:
                return output

            for key, curve in learning_curves.items():
                metric_curve = CurveMetric(xs=curve.x,
                                            ys=curve.y, name=key)
                if key == metric_key:
                    best_acc = max(curve.y)
                visualization_info = LineChartInfo(name=key, x_axis_label="Timestamp", y_axis_label=key)
                output.append(LineMetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

            return output, best_acc
