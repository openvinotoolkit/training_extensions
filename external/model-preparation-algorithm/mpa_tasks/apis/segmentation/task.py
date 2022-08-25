# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import io
import os
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
from mmcv.utils import ConfigDict
from segmentation_tasks.apis.segmentation.config_utils import remove_from_config
from segmentation_tasks.apis.segmentation.ote_utils import TrainingProgressCallback, InferenceProgressCallback
from segmentation_tasks.extension.utils.hooks import OTELoggerHook
from mpa import MPAConstants
from mpa_tasks.apis import BaseTask, TrainType
from mpa_tasks.apis.segmentation import SegmentationConfig
from mpa_tasks.utils.data_utils import get_actmap
from mpa.utils.config_utils import MPAConfig
from mpa.utils.logger import get_logger
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.inference_parameters import default_progress_callback as default_infer_progress_callback
from ote_sdk.entities.label import Domain
from ote_sdk.entities.metrics import (CurveMetric, InfoMetric, LineChartInfo,
                                      MetricsGroup, Performance, ScoreMetric,
                                      VisualizationInfo, VisualizationType)
from ote_sdk.entities.model import (ModelEntity, ModelFormat,
                                    ModelOptimizationType, ModelPrecision)
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.tensor import TensorEntity
from ote_sdk.entities.train_parameters import TrainParameters, default_progress_callback
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import \
    IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.export_interface import (ExportType,
                                                                IExportTask)
from ote_sdk.usecases.tasks.interfaces.inference_interface import \
    IInferenceTask
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from ote_sdk.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
    create_hard_prediction_from_soft_prediction)

from segmentation_tasks.apis.segmentation import OTESegmentationNNCFTask
from ote_sdk.utils.argument_checks import check_input_parameters_type
from ote_sdk.entities.model_template import parse_model_template


logger = get_logger()

TASK_CONFIG = SegmentationConfig


class SegmentationInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IUnload):
    def __init__(self, task_environment: TaskEnvironment):
        # self._should_stop = False
        self.freeze = True
        self.metric = 'mDice'
        super().__init__(TASK_CONFIG, task_environment)

    def infer(self,
              dataset: DatasetEntity,
              inference_parameters: Optional[InferenceParameters] = None
              ) -> DatasetEntity:
        logger.info('infer()')
        dump_features = True
        dump_saliency_map = not inference_parameters.is_evaluation if inference_parameters else True

        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            is_evaluation = inference_parameters.is_evaluation
        else:
            update_progress_callback = default_infer_progress_callback
            is_evaluation = False

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        stage_module = 'SegInferrer'
        self._data_cfg = self._init_test_data_cfg(dataset)
        self._label_dictionary = dict(enumerate(self._labels, 1))
        results = self._run_task(stage_module, mode='train', dataset=dataset, dump_features=dump_features,
                                  dump_saliency_map=dump_saliency_map)
        logger.debug(f'result of run_task {stage_module} module = {results}')
        predictions = results['outputs']
        prediction_results = zip(predictions['eval_predictions'], predictions['feature_vectors'],
                                  predictions['saliency_maps'])
        self._add_predictions_to_dataset(prediction_results, dataset, dump_saliency_map=not is_evaluation)
        return dataset

    def evaluate(self,
                 output_result_set: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        logger.info('called evaluate()')

        if evaluation_metric is not None:
            logger.warning(f'Requested to use {evaluation_metric} metric, '
                           'but parameter is ignored. Use mDice instead.')
        logger.info('Computing mDice')
        metrics = MetricsHelper.compute_dice_averaged_over_pixels(
            output_result_set
        )
        logger.info(f"mDice after evaluation: {metrics.overall_dice.value}")
        output_result_set.performance = metrics.get_performance()

    def unload(self):
        """
        Unload the task
        """
        self.finalize()

    def export(self,
               export_type: ExportType,
               output_model: ModelEntity):
        # copied from OTE inference_task.py
        logger.info('Exporting the model')
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f'not supported export type {export_type}')
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        stage_module = 'SegExporter'
        results = self._run_task(stage_module, mode='train', precision='FP32', export=True)
        results = results.get('outputs')
        logger.debug(f'results of run_task = {results}')
        if results is None:
            logger.error(f"error while exporting model {results.get('msg')}")
            # output_model.model_status = ModelStatus.FAILED
        else:
            bin_file = results.get('bin')
            xml_file = results.get('xml')
            if xml_file is None or bin_file is None:
                raise RuntimeError('invalid status of exporting. bin and xml should not be None')
            with open(bin_file, "rb") as f:
                output_model.set_data('openvino.bin', f.read())
            with open(xml_file, "rb") as f:
                output_model.set_data('openvino.xml', f.read())
            output_model.precision = [ModelPrecision.FP32]
            output_model.optimization_methods = self._optimization_methods
            output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
        logger.info('Exporting completed')

    def _init_recipe_hparam(self) -> dict:
        warmup_iters = int(self._hyperparams.learning_parameters.learning_rate_warmup_iters)
        lr_config = ConfigDict(warmup_iters=warmup_iters) if warmup_iters > 0 \
            else ConfigDict(warmup_iters=warmup_iters, warmup=None)
        return ConfigDict(
            optimizer=ConfigDict(lr=self._hyperparams.learning_parameters.learning_rate),
            lr_config=lr_config,
            data=ConfigDict(
                samples_per_gpu=int(self._hyperparams.learning_parameters.batch_size),
                workers_per_gpu=int(self._hyperparams.learning_parameters.num_workers),
            ),
            runner=ConfigDict(max_epochs=int(self._hyperparams.learning_parameters.num_iters)),
        )

    def _init_recipe(self):
        logger.info('called _init_recipe()')

        recipe_root = os.path.join(MPAConstants.RECIPES_PATH, 'stages/segmentation')
        train_type = self._hyperparams.algo_backend.train_type
        logger.info(f'train type = {train_type}')

        recipe = os.path.join(recipe_root, 'class_incr.py')
        if train_type == TrainType.SemiSupervised:
            recipe = os.path.join(recipe_root, 'cutmix_seg.py')
        elif train_type == TrainType.SelfSupervised:
            # recipe = os.path.join(recipe_root, 'pretrain.yaml')
            raise NotImplementedError(f'train type {train_type} is not implemented yet.')
        elif train_type == TrainType.Incremental:
            recipe = os.path.join(recipe_root, 'class_incr.py')
        else:
            # raise NotImplementedError(f'train type {train_type} is not implemented yet.')
            # FIXME: Temporary remedy for CVS-88098
            logger.warning(f'train type {train_type} is not implemented yet.')

        self._recipe_cfg = MPAConfig.fromfile(recipe)
        self._patch_datasets(self._recipe_cfg)  # for OTE compatibility
        self._patch_evaluation(self._recipe_cfg)  # for OTE compatibility
        self.metric = self._recipe_cfg.evaluation.metric
        if not self.freeze:
            remove_from_config(self._recipe_cfg, 'params_config')
        logger.info(f'initialized recipe = {recipe}')

    def _init_model_cfg(self):
        base_dir = os.path.abspath(os.path.dirname(self.template_file_path))
        return MPAConfig.fromfile(os.path.join(base_dir, 'model.py'))

    def _init_test_data_cfg(self, dataset: DatasetEntity):
        data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    dataset=ConfigDict(
                        ote_dataset=None,
                        labels=self._labels,
                    )
                ),
                test=ConfigDict(
                    ote_dataset=dataset,
                    labels=self._labels,
                )
            )
        )
        return data_cfg

    def _add_predictions_to_dataset(self, prediction_results, dataset, dump_saliency_map):
        """ Loop over dataset again to assign predictions. Convert from MMSegmentation format to OTE format. """

        for dataset_item, (prediction, feature_vector, saliency_map) in zip(dataset, prediction_results):
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

            if dump_saliency_map and saliency_map is not None:
                saliency_map = get_actmap(saliency_map, (dataset_item.width, dataset_item.height) )
                saliency_map_media = ResultMediaEntity(name="saliency_map", type="Saliency map",
                                                annotation_scene=dataset_item.annotation_scene,
                                                numpy=saliency_map, roi=dataset_item.roi)
                dataset_item.append_metadata_item(saliency_map_media, model=self._task_environment.model)

    @staticmethod
    def _patch_datasets(config: MPAConfig, domain=Domain.SEGMENTATION):
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
            cfg.type = 'MPASegIncrDataset'
            cfg.domain = domain
            cfg.ote_dataset = None
            cfg.labels = None
            remove_from_config(cfg, 'ann_dir')
            remove_from_config(cfg, 'img_dir')
            remove_from_config(cfg, 'data_root')
            remove_from_config(cfg, 'split')
            remove_from_config(cfg, 'classes')

            for pipeline_step in cfg.pipeline:
                if pipeline_step.type == 'LoadImageFromFile':
                    pipeline_step.type = 'LoadImageFromOTEDataset'
                elif pipeline_step.type == 'LoadAnnotations':
                    pipeline_step.type = 'LoadAnnotationFromOTEDataset'
                    pipeline_step.domain = domain
                if subset == 'train' and pipeline_step.type == 'Collect':
                    pipeline_step = BaseTask._get_meta_keys(pipeline_step)
            patch_color_conversion(cfg.pipeline)

    @staticmethod
    def _patch_evaluation(config: MPAConfig):
        cfg = config.evaluation
        cfg.pop('classwise', None)
        cfg.metric = 'mDice'
        cfg.save_best = 'mDice'
        cfg.rule = 'greater'
        # EarlyStoppingHook
        for cfg in config.get('custom_hooks', []):
            if 'EarlyStoppingHook' in cfg.type:
                cfg.metric = 'mDice'


class SegmentationTrainTask(SegmentationInferenceTask, ITrainingTask):
    def save_model(self, output_model: ModelEntity):
        logger.info(f'called save_model: {self._model_ckpt}')
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {'model': model_ckpt['state_dict'], 'config': hyperparams_str, 'labels': labels, 'VERSION': 1}

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
        # stop_training_filepath = os.path.join(self._training_work_dir, '.stop_training')
        # open(stop_training_filepath, 'a').close()
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
        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback
        self._time_monitor = TrainingProgressCallback(update_progress_callback)
        self._learning_curves = defaultdict(OTELoggerHook.Curve)

        # learning_curves = defaultdict(OTELoggerHook.Curve)
        stage_module = 'SegTrainer'
        self._data_cfg = self._init_train_data_cfg(dataset)
        self._is_training = True
        results = self._run_task(stage_module, mode='train', dataset=dataset, parameters=train_parameters)

        # Check for stop signal when training has stopped.
        # If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info('Training cancelled.')
            self._should_stop = False
            self._is_training = False
            return

        # get output model
        model_ckpt = results.get('final_ckpt')
        if model_ckpt is None:
            logger.error('cannot find final checkpoint from the results.')
            # output_model.model_status = ModelStatus.FAILED
            return
        else:
            # update checkpoint to the newly trained model
            self._model_ckpt = model_ckpt

        # Get training metrics group from learning curves
        training_metrics, best_score = self._generate_training_metrics_group(self._learning_curves)
        performance = Performance(score=ScoreMetric(value=best_score, name=self.metric),
                                  dashboard_metrics=training_metrics)

        logger.info(f'Final model performance: {str(performance)}')
        # save resulting model
        self.save_model(output_model)
        output_model.performance = performance
        # output_model.model_status = ModelStatus.SUCCESS
        self._is_training = False
        logger.info('train done.')

    def _init_train_data_cfg(self, dataset: DatasetEntity):
        logger.info('init data cfg.')
        data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    dataset=ConfigDict(
                        ote_dataset=dataset.get_subset(Subset.TRAINING),
                        labels=self._labels,
                    )
                ),
                val=ConfigDict(
                    ote_dataset=dataset.get_subset(Subset.VALIDATION),
                    labels=self._labels,
                ),
            )
        )

        # Temparory remedy for cfg.pretty_text error
        for label in self._labels:
            label.hotkey = 'a'
        return data_cfg

    def _generate_training_metrics_group(self, learning_curves) -> Optional[List[MetricsGroup]]:
        """
        Parses the mmsegmentation logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []
        # Model architecture
        architecture = InfoMetric(name='Model architecture', value=self._model_name)
        visualization_info_architecture = VisualizationInfo(name="Model architecture",
                                                            visualisation_type=VisualizationType.TEXT)
        output.append(MetricsGroup(metrics=[architecture],
                                   visualization_info=visualization_info_architecture))
        # Learning curves
        best_score = -1
        for key, curve in learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            if key == f'val/{self.metric}':
                best_score = max(curve.y)
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(MetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))
        return output, best_score


class SegmentationNNCFTask(OTESegmentationNNCFTask):

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for compressing segmentation models using NNCF.
        """
        curr_model_path = task_environment.model_template.model_template_path
        base_model_path = os.path.join(
            os.path.dirname(os.path.abspath(curr_model_path)),
            task_environment.model_template.base_model_path
        )
        if os.path.isfile(base_model_path):
            logger.info(f'Base model for NNCF: {base_model_path}')
            # Redirect to base model
            task_environment.model_template = parse_model_template(base_model_path)
        super().__init__(task_environment)
