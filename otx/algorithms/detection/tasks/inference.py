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
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util
from mmcv.utils import ConfigDict

from otx.algorithms.common.adapters.mmcv.utils import (
    get_configs_by_keys,
    patch_data_pipeline,
    patch_default_config,
    patch_runner,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.common.configs.training_base import TrainType
from otx.algorithms.common.tasks.training_base import BaseTask
from otx.algorithms.common.utils.callback import InferenceProgressCallback
from otx.algorithms.common.utils.ir import embed_ir_model_data
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.utils import (
    patch_datasets,
    patch_evaluation,
)
from otx.algorithms.detection.adapters.mmdet.utils.builder import build_detector
from otx.algorithms.detection.adapters.mmdet.utils.config_utils import (
    cluster_anchors,
    should_cluster_anchors,
)
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.utils import get_det_model_api_configuration
from otx.api.configuration.helper.utils import config_to_bytes
from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from otx.api.entities.model_template import TaskType
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
from otx.api.usecases.tasks.interfaces.explain_interface import IExplainTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from otx.api.utils.dataset_utils import add_saliency_maps_to_dataset_item

logger = get_logger()

RECIPE_TRAIN_TYPE = {
    TrainType.Semisupervised: "semisl.py",
    TrainType.Incremental: "incremental.py",
}


# pylint: disable=too-many-locals, too-many-instance-attributes
class DetectionInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IExplainTask, IUnload):
    """Inference Task Implementation of OTX Detection."""

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment, **kwargs):
        # self._should_stop = False
        super().__init__(DetectionConfig, task_environment, **kwargs)
        self.template_dir = os.path.abspath(os.path.dirname(self.template_file_path))

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main infer function of OTX Detection."""
        logger.info("infer()")

        update_progress_callback = default_progress_callback
        process_saliency_maps = False
        explain_predicted_classes = True
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress  # type: ignore
            process_saliency_maps = inference_parameters.process_saliency_maps
            explain_predicted_classes = inference_parameters.explain_predicted_classes

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)
        # If confidence threshold is adaptive then up-to-date value should be stored in the model
        # and should not be changed during inference. Otherwise user-specified value should be taken.
        if not self._hyperparams.postprocessing.result_based_confidence_threshold:
            self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
        logger.info(f"Confidence threshold {self.confidence_threshold}")

        prediction_results, _ = self._infer_detector(dataset, inference_parameters)
        self._add_predictions_to_dataset(
            prediction_results, dataset, self.confidence_threshold, process_saliency_maps, explain_predicted_classes
        )
        logger.info("Inference completed")
        return dataset

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main explain function of OTX Detection."""
        logger.info("explain()")

        update_progress_callback = default_progress_callback
        process_saliency_maps = False
        explain_predicted_classes = True
        if explain_parameters is not None:
            update_progress_callback = explain_parameters.update_progress  # type: ignore
            process_saliency_maps = explain_parameters.process_saliency_maps
            explain_predicted_classes = explain_parameters.explain_predicted_classes

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)
        detections, explain_results = self._explain_detector(dataset, explain_parameters)
        self._add_explanations_to_dataset(
            detections, explain_results, dataset, process_saliency_maps, explain_predicted_classes
        )
        logger.info("Explain completed")
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

    def _explain_detector(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[InferenceParameters] = None,
    ) -> Tuple[List[List[np.array]], List[np.array]]:
        """Run explain stage and return detections and saliency maps."""

        stage_module = "DetectionExplainer"
        self._data_cfg = self._init_test_data_cfg(dataset)
        results = self._run_task(
            stage_module,
            mode="train",
            dataset=dataset,
            explainer=explain_parameters.explainer if explain_parameters else None,
        )
        detections = results["outputs"]["detections"]
        explain_results = results["outputs"]["saliency_maps"]
        return detections, explain_results

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
        self.cleanup()

    @check_input_parameters_type()
    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = False,
    ):
        """Export function of OTX Detection Task."""
        # copied from OTX inference_task.py
        logger.info("Exporting the model")
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f"not supported export type {export_type}")
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        stage_module = "DetectionExporter"
        results = self._run_task(
            stage_module,
            mode="train",
            export=True,
            dump_features=dump_features,
            enable_fp16=(precision == ModelPrecision.FP16),
        )
        outputs = results.get("outputs")
        logger.debug(f"results of run_task = {outputs}")
        if outputs is None:
            raise RuntimeError(results.get("msg"))

        bin_file = outputs.get("bin")
        xml_file = outputs.get("xml")

        ir_extra_data = get_det_model_api_configuration(
            self._task_environment.label_schema, self._task_type, self.confidence_threshold
        )
        embed_ir_model_data(xml_file, ir_extra_data)

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
        output_model.set_data("config.json", config_to_bytes(self._hyperparams))
        output_model.precision = self._precision
        output_model.optimization_methods = self._optimization_methods
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        logger.info("Exporting completed")

    def _init_recipe_hparam(self) -> dict:
        configs = super()._init_recipe_hparam()
        # Update tiling parameters if tiling is enabled
        if bool(self._hyperparams.tiling_parameters.enable_tiling):
            logger.info("Tiling Enabled")
            tiling_params = ConfigDict(
                tile_size=int(self._hyperparams.tiling_parameters.tile_size),
                overlap_ratio=float(self._hyperparams.tiling_parameters.tile_overlap),
                max_per_img=int(self._hyperparams.tiling_parameters.tile_max_number),
            )
            configs.update(
                ConfigDict(
                    data=ConfigDict(
                        train=tiling_params,
                        val=tiling_params,
                        test=tiling_params,
                    )
                )
            )
            configs.update(dict(evaluation=dict(iou_thr=[0.5])))

        configs["use_adaptive_interval"] = self._hyperparams.learning_parameters.use_adaptive_interval
        return configs

    def _init_recipe(self):
        logger.info("called _init_recipe()")

        self._recipe_cfg = self._init_model_cfg()

        options_for_patch_datasets = {"type": "OTXDetDataset"}
        patch_default_config(self._recipe_cfg)
        patch_runner(self._recipe_cfg)
        patch_data_pipeline(self._recipe_cfg, self.data_pipeline_path)
        patch_datasets(
            self._recipe_cfg,
            self._task_type.domain,
            **options_for_patch_datasets,
        )  # for OTX compatibility
        patch_evaluation(self._recipe_cfg)  # for OTX compatibility

    def _init_model_cfg(self):
        model_cfg = MPAConfig.fromfile(os.path.join(self._model_dir, "model.py"))
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

    def _update_stage_module(self, stage_module):
        module_prefix = {TrainType.Incremental: "Incr", TrainType.Semisupervised: "SemiSL"}
        if self._train_type == TrainType.Semisupervised and stage_module == "DetectionExporter":
            stage_module = "SemiSLDetectionExporter"
        elif self._train_type in module_prefix and stage_module in [
            "DetectionTrainer",
            "DetectionInferrer",
        ]:
            stage_module = module_prefix[self._train_type] + stage_module
        return stage_module

    def _get_shapes(self, all_results, width, height, confidence_threshold):
        if self._task_type == TaskType.DETECTION:
            shapes = self._det_add_predictions_to_dataset(all_results, width, height, confidence_threshold)
        elif self._task_type in {
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.ROTATED_DETECTION,
        }:
            shapes = self._ins_seg_add_predictions_to_dataset(all_results, width, height, confidence_threshold)
        else:
            raise RuntimeError(f"MPA results assignment not implemented for task: {self._task_type}")
        return shapes

    def _add_predictions_to_dataset(
        self,
        prediction_results,
        dataset,
        confidence_threshold=0.0,
        process_saliency_maps=False,
        explain_predicted_classes=True,
    ):
        """Loop over dataset again to assign predictions. Convert from MMDetection format to OTX format."""
        for dataset_item, (all_results, feature_vector, saliency_map) in zip(dataset, prediction_results):
            shapes = self._get_shapes(all_results, dataset_item.width, dataset_item.height, confidence_threshold)
            dataset_item.append_annotations(shapes)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if saliency_map is not None:
                labels = self._labels.copy()
                if saliency_map.shape[0] == len(labels) + 1:
                    # Include the background as the last category
                    labels.append(LabelEntity("background", Domain.DETECTION))

                predicted_scored_labels = []
                for shape in shapes:
                    predicted_scored_labels += shape.get_labels()

                add_saliency_maps_to_dataset_item(
                    dataset_item=dataset_item,
                    saliency_map=saliency_map,
                    model=self._task_environment.model,
                    labels=labels,
                    predicted_scored_labels=predicted_scored_labels,
                    explain_predicted_classes=explain_predicted_classes,
                    process_saliency_maps=process_saliency_maps,
                )

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
                if isinstance(mask, dict):
                    mask = mask_util.decode(mask)
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

    def _add_explanations_to_dataset(
        self, detections, explain_results, dataset, process_saliency_maps, explain_predicted_classes
    ):
        """Add saliency map to the dataset."""
        for dataset_item, detection, saliency_map in zip(dataset, detections, explain_results):
            labels = self._labels.copy()
            if saliency_map.shape[0] == len(labels) + 1:
                # Include the background as the last category
                labels.append(LabelEntity("background", Domain.DETECTION))

            shapes = self._get_shapes(detection, dataset_item.width, dataset_item.height, 0.4)
            predicted_scored_labels = []
            for shape in shapes:
                predicted_scored_labels += shape.get_labels()

            add_saliency_maps_to_dataset_item(
                dataset_item=dataset_item,
                saliency_map=saliency_map,
                model=self._task_environment.model,
                labels=labels,
                predicted_scored_labels=predicted_scored_labels,
                explain_predicted_classes=explain_predicted_classes,
                process_saliency_maps=process_saliency_maps,
            )

    @staticmethod
    def _update_anchors(origin, new):
        logger.info("Updating anchors")
        origin["heights"] = new["heights"]
        origin["widths"] = new["widths"]

    def _initialize_post_hook(self, options=None):
        super()._initialize_post_hook(options)
        options["model_builder"] = build_detector

        # if self._anchors are set somewhere, anchors had already been clusted
        # by this method or by loading trained model
        if should_cluster_anchors(self._recipe_cfg) and len(self._anchors) == 0:
            otx_dataset = get_configs_by_keys(self._data_cfg.data.train, "otx_dataset")
            assert len(otx_dataset) == 1
            otx_dataset = otx_dataset[0]
            cluster_anchors(
                self._recipe_cfg,
                otx_dataset,
            )
            self._update_anchors(self._anchors, self._recipe_cfg.model.bbox_head.anchor_generator)
