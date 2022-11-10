"""Inference Task of OTX Detection."""

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

import os
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from mmcv.utils import ConfigDict
from mpa import MPAConstants
from mpa.utils.config_utils import MPAConfig
from mpa.utils.logger import get_logger

from otx.algorithms.common.configs.training_base import TrainType
from otx.algorithms.common.tasks.training_base import BaseTask
from otx.algorithms.common.utils.callback import InferenceProgressCallback
from otx.algorithms.detection.adapters.mmdet.utils import (
    patch_data_pipeline,
    patch_datasets,
    patch_evaluation,
)
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import Domain
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from otx.api.entities.model_template import TaskType
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.entities.train_parameters import default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from otx.api.utils.vis_utils import get_actmap

logger = get_logger()


# pylint: disable=too-many-locals
class DetectionInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IUnload):
    """Inference Task Implementation of OTX Detection."""

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        # self._should_stop = False
        super().__init__(DetectionConfig, task_environment)

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main infer function of OTX Detection."""
        logger.info("infer()")

        if inference_parameters:
            update_progress_callback = inference_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)
        # If confidence threshold is adaptive then up-to-date value should be stored in the model
        # and should not be changed during inference. Otherwise user-specified value should be taken.
        if not self._hyperparams.postprocessing.result_based_confidence_threshold:
            self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
        logger.info(f"Confidence threshold {self.confidence_threshold}")

        prediction_results, _ = self._infer_detector(dataset, inference_parameters)
        self._add_predictions_to_dataset(prediction_results, dataset, self.confidence_threshold)
        logger.info("Inference completed")
        return dataset

    def _infer_detector(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> Tuple[Iterable, float]:
        """Inference wrapper.

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
        stage_module = "DetectionInferrer"
        self._data_cfg = self._init_test_data_cfg(dataset)
        dump_features = True
        dump_saliency_map = not inference_parameters.is_evaluation if inference_parameters else True
        results = self._run_task(
            stage_module,
            mode="train",
            dataset=dataset,
            eval=inference_parameters.is_evaluation if inference_parameters else False,
            dump_features=dump_features,
            dump_saliency_map=dump_saliency_map,
        )
        # TODO: InferenceProgressCallback register
        logger.debug(f"result of run_task {stage_module} module = {results}")
        output = results["outputs"]
        metric = output["metric"]
        predictions = output["detections"]
        assert len(output["detections"]) == len(output["feature_vectors"]) == len(output["saliency_maps"]), (
            "Number of elements should be the same, however, number of outputs are "
            f"{len(output['detections'])}, {len(output['feature_vectors'])}, and {len(output['saliency_maps'])}"
        )
        prediction_results = zip(predictions, output["feature_vectors"], output["saliency_maps"])
        return prediction_results, metric

    @check_input_parameters_type()
    def evaluate(
        self,
        output_resultset: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        """Evaluate function of OTX Detection Task."""
        logger.info("called evaluate()")
        if evaluation_metric is not None:
            logger.warning(
                f"Requested to use {evaluation_metric} metric, " "but parameter is ignored. Use F-measure instead."
            )
        metric = MetricsHelper.compute_f_measure(output_resultset)
        logger.info(f"F-measure after evaluation: {metric.f_measure.value}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    def unload(self):
        """Unload the task."""
        self._delete_scratch_space()

    @check_input_parameters_type()
    def export(self, export_type: ExportType, output_model: ModelEntity):
        """Export function of OTX Detection Task."""
        # copied from OTX inference_task.py
        logger.info("Exporting the model")
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f"not supported export type {export_type}")
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        stage_module = "DetectionExporter"
        results = self._run_task(stage_module, mode="train", precision="FP32", export=True)
        results = results.get("outputs")
        logger.debug(f"results of run_task = {results}")
        if results is None:
            logger.error("error while exporting model, result is None")
        else:
            bin_file = results.get("bin")
            xml_file = results.get("xml")
            if xml_file is None or bin_file is None:
                raise RuntimeError("invalid status of exporting. bin and xml should not be None")
            with open(bin_file, "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            with open(xml_file, "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            output_model.set_data(
                "confidence_threshold",
                np.array([self.confidence_threshold], dtype=np.float32).tobytes(),
            )
            output_model.precision = [ModelPrecision.FP32]
            output_model.optimization_methods = self._optimization_methods
            output_model.set_data(
                "label_schema.json",
                label_schema_to_bytes(self._task_environment.label_schema),
            )
        logger.info("Exporting completed")

    def _init_recipe_hparam(self) -> dict:
        configs = super()._init_recipe_hparam()
        configs["use_adaptive_interval"] = self._hyperparams.learning_parameters.use_adaptive_interval
        return configs

    def _init_recipe(self):
        logger.info("called _init_recipe()")

        recipe_root = os.path.join(MPAConstants.RECIPES_PATH, "stages/detection")
        if self._task_type.domain in {
            Domain.INSTANCE_SEGMENTATION,
            Domain.ROTATED_DETECTION,
        }:
            recipe_root = os.path.join(MPAConstants.RECIPES_PATH, "stages/instance-segmentation")

        train_type = self._hyperparams.algo_backend.train_type
        logger.info(f"train type = {train_type}")

        if self._data_cfg.get('data', None):
            if self._data_cfg.data.get('unlabeled', None):
                train_type = TrainType.SEMISUPERVISED
                logger.info(f"Unlabeled data detected - convert to {train_type} mode...")
        recipe = os.path.join(recipe_root, "imbalance.py")
        if train_type == TrainType.SEMISUPERVISED:
            recipe = os.path.join(recipe_root, "unbiased_teacher.py")
        elif train_type == TrainType.SELFSUPERVISED:
            # recipe = os.path.join(recipe_root, 'pretrain.yaml')
            raise NotImplementedError(f"train type {train_type} is not implemented yet.")
        elif train_type == TrainType.INCREMENTAL:
            recipe = os.path.join(recipe_root, "imbalance.py")
        else:
            # raise NotImplementedError(f'train type {train_type} is not implemented yet.')
            # FIXME: Temporary remedy for CVS-88098
            logger.warning(f"train type {train_type} is not implemented yet.")
        logger.info(f"train type = {train_type, recipe}")

        self._recipe_cfg = MPAConfig.fromfile(recipe)
        patch_data_pipeline(self._recipe_cfg, self.template_file_path)
        patch_datasets(self._recipe_cfg, self._task_type.domain)  # for OTX compatibility
        patch_evaluation(self._recipe_cfg)  # for OTX compatibility
        logger.info(f"initialized recipe = {recipe}")

    def _init_model_cfg(self):
        base_dir = os.path.abspath(os.path.dirname(self.template_file_path))
        model_cfg = MPAConfig.fromfile(os.path.join(base_dir, "model.py"))
        if len(self._anchors) != 0:
            self._update_anchors(model_cfg.model.bbox_head.anchor_generator, self._anchors)
        return model_cfg

    def _init_test_data_cfg(self, dataset: DatasetEntity):
        data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    otx_dataset=None,
                    labels=self._labels,
                ),
                test=ConfigDict(
                    otx_dataset=dataset,
                    labels=self._labels,
                ),
            )
        )
        return data_cfg

    def _add_predictions_to_dataset(self, prediction_results, dataset, confidence_threshold=0.0):
        """Loop over dataset again to assign predictions. Convert from MMDetection format to OTX format."""
        for dataset_item, (all_results, feature_vector, saliency_map) in zip(dataset, prediction_results):
            width = dataset_item.width
            height = dataset_item.height

            shapes = []
            if self._task_type == TaskType.DETECTION:
                shapes = self._det_add_predictions_to_dataset(all_results, width, height, confidence_threshold)
            elif self._task_type in {
                TaskType.INSTANCE_SEGMENTATION,
                TaskType.ROTATED_DETECTION,
            }:
                shapes = self._ins_seg_add_predictions_to_dataset(all_results, width, height, confidence_threshold)
            else:
                raise RuntimeError(f"MPA results assignment not implemented for task: {self._task_type}")

            dataset_item.append_annotations(shapes)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if saliency_map is not None:
                saliency_map = get_actmap(saliency_map, (dataset_item.width, dataset_item.height))
                saliency_map_media = ResultMediaEntity(
                    name="Saliency Map",
                    type="saliency_map",
                    annotation_scene=dataset_item.annotation_scene,
                    numpy=saliency_map,
                    roi=dataset_item.roi,
                )
                dataset_item.append_metadata_item(saliency_map_media, model=self._task_environment.model)

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

                assigned_label = [ScoredLabel(self._labels[label_idx], probability=probability)]
                if coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0:
                    continue

                shapes.append(
                    Annotation(
                        Rectangle(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                        labels=assigned_label,
                    )
                )
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
        origin["heights"] = new["heights"]
        origin["widths"] = new["widths"]
