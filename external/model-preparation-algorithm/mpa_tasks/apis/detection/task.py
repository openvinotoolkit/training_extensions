# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import io
import os
from collections import defaultdict
from typing import List, Optional, Tuple, Iterable

import cv2
import numpy as np
import torch
from mmcv.utils import ConfigDict
from detection_tasks.apis.detection.config_utils import remove_from_config
from detection_tasks.apis.detection.ote_utils import TrainingProgressCallback, InferenceProgressCallback
from detection_tasks.extension.utils.hooks import OTELoggerHook
from mpa_tasks.apis import BaseTask, TrainType
from mpa_tasks.apis.detection import DetectionConfig
from mpa import MPAConstants
from mpa.utils.config_utils import MPAConfig
from mpa.utils.logger import get_logger
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label import Domain
from ote_sdk.entities.metrics import (BarChartInfo, BarMetricsGroup,
                                      CurveMetric, LineChartInfo,
                                      LineMetricsGroup, MetricsGroup,
                                      ScoreMetric, VisualizationType)
from ote_sdk.entities.model import (ModelEntity, ModelFormat,
                                    ModelPrecision, ModelOptimizationType)
from ote_sdk.entities.model_template import TaskType
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.polygon import Point, Polygon
from ote_sdk.entities.shapes.rectangle import Rectangle
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

from detection_tasks.apis.detection import OTEDetectionNNCFTask
from ote_sdk.utils.argument_checks import check_input_parameters_type
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.utils.vis_utils import get_actmap


logger = get_logger()

TASK_CONFIG = DetectionConfig


class DetectionInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IUnload):
    def __init__(self, task_environment: TaskEnvironment):
        # self._should_stop = False
        super().__init__(TASK_CONFIG, task_environment)

    def infer(self,
              dataset: DatasetEntity,
              inference_parameters: Optional[InferenceParameters] = None
              ) -> DatasetEntity:
        logger.info('infer()')

        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)
        # If confidence threshold is adaptive then up-to-date value should be stored in the model
        # and should not be changed during inference. Otherwise user-specified value should be taken.
        if not self._hyperparams.postprocessing.result_based_confidence_threshold:
            self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
        logger.info(f'Confidence threshold {self.confidence_threshold}')

        prediction_results, _ = self._infer_detector(dataset, inference_parameters)
        self._add_predictions_to_dataset(prediction_results, dataset, self.confidence_threshold)
        logger.info('Inference completed')
        return dataset

    def _infer_detector(self, dataset: DatasetEntity,
                        inference_parameters: Optional[InferenceParameters] = None) -> Tuple[Iterable, float]:
        """ Inference wrapper

        This method triggers the inference and returns `prediction_results` zipped with prediction results,
        feature vectors, and saliency maps. `metric` is returned as a float value if InferenceParameters.is_evaluation
        is set to true, otherwise, `None` is returned.

        Args:
            dataset (DatasetEntity): the validation or test dataset to be inferred with
            inference_parameters (Optional[InferenceParameters], optional): Option to run evaluation or not.
                If `InferenceParameters.is_evaluation=True` then metric is returned, otherwise, both metric and
                saliency maps are empty. Defaults to None.

        Returns:
            Tuple[Iterable, float]: Iterable prediction results for each sample and metric for on the given dataset
        """
        stage_module = 'DetectionInferrer'
        self._data_cfg = self._init_test_data_cfg(dataset)
        dump_features = True
        dump_saliency_map = not inference_parameters.is_evaluation if inference_parameters else True
        results = self._run_task(stage_module,
                                 mode='train',
                                 dataset=dataset,
                                 eval=inference_parameters.is_evaluation if inference_parameters else False,
                                 dump_features=dump_features,
                                 dump_saliency_map=dump_saliency_map)
        # TODO: InferenceProgressCallback register
        logger.debug(f'result of run_task {stage_module} module = {results}')
        output = results['outputs']
        metric = output['metric']
        predictions = output['detections']
        assert len(output['detections']) == len(output['feature_vectors']) == len(output['saliency_maps']), \
               'Number of elements should be the same, however, number of outputs are ' \
               f"{len(output['detections'])}, {len(output['feature_vectors'])}, and {len(output['saliency_maps'])}"
        prediction_results = zip(predictions, output['feature_vectors'], output['saliency_maps'])
        return prediction_results, metric

    def evaluate(self,
                 output_result_set: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        logger.info('called evaluate()')
        if evaluation_metric is not None:
            logger.warning(f'Requested to use {evaluation_metric} metric, '
                           'but parameter is ignored. Use F-measure instead.')
        metric = MetricsHelper.compute_f_measure(output_result_set)
        logger.info(f"F-measure after evaluation: {metric.f_measure.value}")
        output_result_set.performance = metric.get_performance()
        logger.info('Evaluation completed')

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

        stage_module = 'DetectionExporter'
        results = self._run_task(stage_module, mode='train', precision='FP32', export=True)
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
            output_model.set_data(
                'confidence_threshold',
                np.array([self.confidence_threshold], dtype=np.float32).tobytes())
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

        recipe_root = os.path.join(MPAConstants.RECIPES_PATH, 'stages/detection')
        if self._task_type.domain in {Domain.INSTANCE_SEGMENTATION, Domain.ROTATED_DETECTION}:
            recipe_root = os.path.join(MPAConstants.RECIPES_PATH, 'stages/instance-segmentation')

        train_type = self._hyperparams.algo_backend.train_type
        logger.info(f'train type = {train_type}')

        recipe = os.path.join(recipe_root, 'imbalance.py')
        if train_type == TrainType.SemiSupervised:
            recipe = os.path.join(recipe_root, 'unbiased_teacher.py')
        elif train_type == TrainType.SelfSupervised:
            # recipe = os.path.join(recipe_root, 'pretrain.yaml')
            raise NotImplementedError(f'train type {train_type} is not implemented yet.')
        elif train_type == TrainType.Incremental:
            recipe = os.path.join(recipe_root, 'imbalance.py')
        else:
            # raise NotImplementedError(f'train type {train_type} is not implemented yet.')
            # FIXME: Temporary remedy for CVS-88098
            logger.warning(f'train type {train_type} is not implemented yet.')

        self._recipe_cfg = MPAConfig.fromfile(recipe)
        self._patch_data_pipeline()
        self._patch_datasets(self._recipe_cfg, self._task_type.domain)  # for OTE compatibility
        self._patch_evaluation(self._recipe_cfg)  # for OTE compatibility
        logger.info(f'initialized recipe = {recipe}')

    def _init_model_cfg(self):
        base_dir = os.path.abspath(os.path.dirname(self.template_file_path))
        model_cfg = MPAConfig.fromfile(os.path.join(base_dir, 'model.py'))
        if len(self._anchors) != 0:
            self._update_anchors(model_cfg.model.bbox_head.anchor_generator, self._anchors)
        return model_cfg

    def _init_test_data_cfg(self, dataset: DatasetEntity):
        data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    ote_dataset=None,
                    labels=self._labels,
                ),
                test=ConfigDict(
                    ote_dataset=dataset,
                    labels=self._labels,
                )
            )
        )
        return data_cfg

    def _add_predictions_to_dataset(self, prediction_results, dataset, confidence_threshold=0.0):
        """ Loop over dataset again to assign predictions. Convert from MMDetection format to OTE format. """
        for dataset_item, (all_results, feature_vector, saliency_map) in zip(dataset, prediction_results):
            width = dataset_item.width
            height = dataset_item.height

            shapes = []
            if self._task_type == TaskType.DETECTION:
                shapes = self._det_add_predictions_to_dataset(all_results, width, height, confidence_threshold)
            elif self._task_type in {TaskType.INSTANCE_SEGMENTATION, TaskType.ROTATED_DETECTION}:
                shapes = self._ins_seg_add_predictions_to_dataset(all_results, width, height, confidence_threshold)
            else:
                raise RuntimeError(
                    f"MPA results assignment not implemented for task: {self._task_type}")

            dataset_item.append_annotations(shapes)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if saliency_map is not None:
                saliency_map = get_actmap(saliency_map, (dataset_item.width, dataset_item.height))
                saliency_map_media = ResultMediaEntity(name="Saliency Map", type="saliency_map",
                                                       annotation_scene=dataset_item.annotation_scene,
                                                       numpy=saliency_map, roi=dataset_item.roi)
                dataset_item.append_metadata_item(saliency_map_media, model=self._task_environment.model)

    def _patch_data_pipeline(self):
        base_dir = os.path.abspath(os.path.dirname(self.template_file_path))
        data_pipeline_path = os.path.join(base_dir, 'data_pipeline.py')
        if os.path.exists(data_pipeline_path):
            data_pipeline_cfg = MPAConfig.fromfile(data_pipeline_path)
            self._recipe_cfg.merge_from_dict(data_pipeline_cfg)

    @staticmethod
    def _patch_datasets(config: MPAConfig, domain=Domain.DETECTION):
        # Copied from ote/apis/detection/config_utils.py
        # Added 'unlabeled' data support

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
        for subset in ('train', 'val', 'test', 'unlabeled'):
            cfg = config.data.get(subset, None)
            if not cfg:
                continue
            if cfg.type == 'RepeatDataset' or cfg.type == 'MultiImageMixDataset':
                cfg = cfg.dataset
            cfg.type = 'MPADetDataset'
            cfg.domain = domain
            cfg.ote_dataset = None
            cfg.labels = None
            remove_from_config(cfg, 'ann_file')
            remove_from_config(cfg, 'img_prefix')
            remove_from_config(cfg, 'classes')  # Get from DatasetEntity
            for pipeline_step in cfg.pipeline:
                if pipeline_step.type == 'LoadImageFromFile':
                    pipeline_step.type = 'LoadImageFromOTEDataset'
                if pipeline_step.type == 'LoadAnnotations':
                    pipeline_step.type = 'LoadAnnotationFromOTEDataset'
                    pipeline_step.domain = domain
                    pipeline_step.min_size = cfg.pop('min_size', -1)
                if subset == 'train' and pipeline_step.type == 'Collect':
                    pipeline_step = BaseTask._get_meta_keys(pipeline_step)
            patch_color_conversion(cfg.pipeline)

    @staticmethod
    def _patch_evaluation(config: MPAConfig):
        cfg = config.evaluation
        # CocoDataset.evaluate -> CustomDataset.evaluate
        cfg.pop('classwise', None)
        cfg.metric = 'mAP'
        cfg.save_best = 'mAP'
        # EarlyStoppingHook
        for cfg in config.get('custom_hooks', []):
            if 'EarlyStoppingHook' in cfg.type:
                cfg.metric = 'mAP'

    def _det_add_predictions_to_dataset(self, all_results, width, height, confidence_threshold):
        shapes = []
        for label_idx, detections in enumerate(all_results):
            for i in range(detections.shape[0]):
                probability = float(detections[i, 4])
                coords = detections[i, :4].astype(float).copy()
                coords /= np.array([width, height, width, height], dtype=float)
                coords = np.clip(coords, 0, 1)

                if probability < confidence_threshold:
                    continue

                assigned_label = [ScoredLabel(self._labels[label_idx],
                                              probability=probability)]
                if coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0:
                    continue

                shapes.append(Annotation(
                    Rectangle(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                    labels=assigned_label))
        return shapes

    def _ins_seg_add_predictions_to_dataset(self, all_results, width, height, confidence_threshold):
        shapes = []
        for label_idx, (boxes, masks) in enumerate(zip(*all_results)):
            for mask, probability in zip(masks, boxes[:, 4]):
                mask = mask.astype(np.uint8)
                probability = float(probability)
                contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchies is None:
                    continue
                for contour, hierarchy in zip(contours, hierarchies[0]):
                    if hierarchy[3] != -1:
                        continue
                    if len(contour) <= 2 or probability < confidence_threshold:
                        continue
                    if self._task_type == TaskType.INSTANCE_SEGMENTATION:
                        points = [Point(x=point[0][0] / width, y=point[0][1] / height) for point in contour]
                    else:
                        box_points = cv2.boxPoints(cv2.minAreaRect(contour))
                        points = [Point(x=point[0] / width, y=point[1] / height) for point in box_points]
                    labels = [ScoredLabel(self._labels[label_idx], probability=probability)]
                    polygon = Polygon(points=points)
                    if cv2.contourArea(contour) > 0 and polygon.get_area() > 1e-12:
                        shapes.append(Annotation(polygon, labels=labels, id=ID(f"{label_idx:08}")))
        return shapes

    @staticmethod
    def _update_anchors(origin, new):
        logger.info("Updating anchors")
        origin['heights'] = new['heights']
        origin['widths'] = new['widths']


class DetectionTrainTask(DetectionInferenceTask, ITrainingTask):
    def save_model(self, output_model: ModelEntity):
        logger.info('called save_model')
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {
            'model': model_ckpt['state_dict'], 'config': hyperparams_str, 'labels': labels,
            'confidence_threshold': self.confidence_threshold, 'VERSION': 1
        }
        if hasattr(self._model_cfg.model, 'bbox_head') and hasattr(self._model_cfg.model.bbox_head, 'anchor_generator'):
            if getattr(self._model_cfg.model.bbox_head.anchor_generator, 'reclustering_anchors', False):
                modelinfo['anchors'] = {}
                self._update_anchors(modelinfo['anchors'], self._model_cfg.model.bbox_head.anchor_generator)

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
        # Check for stop signal when training has stopped.
        # If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info('Training cancelled.')
            self._should_stop = False
            self._is_training = False
            return

        # Set OTE LoggerHook & Time Monitor
        update_progress_callback = default_progress_callback
        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
        self._time_monitor = TrainingProgressCallback(update_progress_callback)
        self._learning_curves = defaultdict(OTELoggerHook.Curve)

        stage_module = 'DetectionTrainer'
        self._data_cfg = self._init_train_data_cfg(dataset)
        self._is_training = True
        results = self._run_task(stage_module, mode='train', dataset=dataset, parameters=train_parameters)

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
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

        # Update anchors
        if hasattr(self._model_cfg.model, 'bbox_head') and hasattr(self._model_cfg.model.bbox_head, 'anchor_generator'):
            if getattr(self._model_cfg.model.bbox_head.anchor_generator, 'reclustering_anchors', False):
                self._update_anchors(self._anchors, self._model_cfg.model.bbox_head.anchor_generator)

        # get prediction on validation set
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        val_preds, val_map = self._infer_detector(val_dataset, InferenceParameters(is_evaluation=True))

        preds_val_dataset = val_dataset.with_empty_annotations()
        self._add_predictions_to_dataset(val_preds, preds_val_dataset, 0.0)

        result_set = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=val_dataset,
            prediction_dataset=preds_val_dataset
        )

        # adjust confidence threshold
        if self._hyperparams.postprocessing.result_based_confidence_threshold:
            logger.info('Adjusting the confidence threshold')
            metric = MetricsHelper.compute_f_measure(result_set, vary_confidence_threshold=True)
            best_confidence_threshold = metric.best_confidence_threshold.value
            if best_confidence_threshold is None:
                raise ValueError("Cannot compute metrics: Invalid confidence threshold!")
            logger.info(f"Setting confidence threshold to {best_confidence_threshold} based on results")
            self.confidence_threshold = best_confidence_threshold
        else:
            metric = MetricsHelper.compute_f_measure(result_set, vary_confidence_threshold=False)

        # compose performance statistics
        performance = metric.get_performance()
        performance.dashboard_metrics.extend(self._generate_training_metrics(self._learning_curves, val_map))
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
                    ote_dataset=dataset.get_subset(Subset.TRAINING),
                    labels=self._labels,
                ),
                val=ConfigDict(
                    ote_dataset=dataset.get_subset(Subset.VALIDATION),
                    labels=self._labels,
                ),
                unlabeled=ConfigDict(
                    ote_dataset=dataset.get_subset(Subset.UNLABELED),
                    labels=self._labels,
                ),
            )
        )
        # Temparory remedy for cfg.pretty_text error
        for label in self._labels:
            label.hotkey = 'a'
        return data_cfg

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


class DetectionNNCFTask(OTEDetectionNNCFTask):

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for compressing detection models using NNCF.
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
