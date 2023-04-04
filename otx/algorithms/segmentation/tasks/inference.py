"""Inference Task of OTX Segmentation."""

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
from typing import Dict, Optional

import numpy as np
from mmcv.utils import ConfigDict

from otx.algorithms.common.adapters.mmcv.utils import (
    patch_data_pipeline,
    patch_default_config,
    patch_runner,
    remove_from_configs_by_type,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.common.configs import TrainType
from otx.algorithms.common.tasks import BaseTask
from otx.algorithms.common.utils.callback import InferenceProgressCallback
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.mmseg.utils.builder import build_segmentor
from otx.algorithms.segmentation.adapters.mmseg.utils.config_utils import (
    patch_datasets,
    patch_evaluation,
)
from otx.algorithms.segmentation.adapters.openvino.model_wrappers.blur import (
    get_activation_map,
)
from otx.algorithms.segmentation.configs.base import SegmentationConfig
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.inference_parameters import (
    default_progress_callback as default_infer_progress_callback,
)
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
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
from otx.api.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
    create_hard_prediction_from_soft_prediction,
)

logger = get_logger()


RECIPE_TRAIN_TYPE = {
    TrainType.Semisupervised: "semisl.py",
    TrainType.Incremental: "incremental.py",
    TrainType.Selfsupervised: "selfsl.py",
}


# pylint: disable=too-many-locals, too-many-instance-attributes, attribute-defined-outside-init
class SegmentationInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IUnload):
    """Inference Task Implementation of OTX Segmentation."""

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment, **kwargs):
        # self._should_stop = False
        self.freeze = True
        self.metric = "mDice"
        self._label_dictionary = {}  # type: Dict

        super().__init__(SegmentationConfig, task_environment, **kwargs)
        self._label_dictionary = dict(enumerate(sorted(self._labels), 1))

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def infer(
        self, dataset: DatasetEntity, inference_parameters: Optional[InferenceParameters] = None
    ) -> DatasetEntity:
        """Main infer function of OTX Segmentation."""
        logger.info("infer()")

        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            is_evaluation = inference_parameters.is_evaluation
        else:
            update_progress_callback = default_infer_progress_callback
            is_evaluation = False

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        stage_module = "SegInferrer"
        self._data_cfg = self._init_test_data_cfg(dataset)

        dump_features = True

        results = self._run_task(
            stage_module,
            mode="train",
            dataset=dataset,
            dump_features=dump_features,
        )
        logger.debug(f"result of run_task {stage_module} module = {results}")
        predictions = results["outputs"]
        prediction_results = zip(predictions["eval_predictions"], predictions["feature_vectors"])
        self._add_predictions_to_dataset(prediction_results, dataset, dump_soft_prediction=not is_evaluation)
        return dataset

    @check_input_parameters_type()
    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate function of OTX Segmentation Task."""
        logger.info("called evaluate()")

        if evaluation_metric is not None:
            logger.warning(
                f"Requested to use {evaluation_metric} metric, " "but parameter is ignored. Use mDice instead."
            )
        logger.info("Computing mDice")
        metrics = MetricsHelper.compute_dice_averaged_over_pixels(output_resultset)
        logger.info(f"mDice after evaluation: {metrics.overall_dice.value}")
        output_resultset.performance = metrics.get_performance()

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
        """Export function of OTX Segmentation Task."""
        logger.info("Exporting the model")
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f"not supported export type {export_type}")
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        stage_module = "SegExporter"
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
        if xml_file is None or bin_file is None:
            raise RuntimeError("invalid status of exporting. bin and xml should not be None")
        with open(bin_file, "rb") as f:
            output_model.set_data("openvino.bin", f.read())
        with open(xml_file, "rb") as f:
            output_model.set_data("openvino.xml", f.read())
        output_model.precision = self._precision
        output_model.optimization_methods = self._optimization_methods
        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
        logger.info("Exporting completed")

    def _init_recipe(self):
        logger.info("called _init_recipe()")
        # TODO: Need to remove the hard coding for supcon only.
        if (
            self._train_type in RECIPE_TRAIN_TYPE
            and self._train_type == TrainType.Incremental
            and self._hyperparams.learning_parameters.enable_supcon
            and not self._model_dir.endswith("supcon")
        ):
            self._model_dir = os.path.join(self._model_dir, "supcon")

        self._recipe_cfg = self._init_model_cfg()
        options_for_patch_datasets = {"type": "MPASegDataset"}
        patch_default_config(self._recipe_cfg)
        patch_runner(self._recipe_cfg)
        patch_data_pipeline(self._recipe_cfg, self.data_pipeline_path)
        patch_datasets(
            self._recipe_cfg,
            self._task_type.domain,
            **options_for_patch_datasets,
        )  # for OTX compatibility
        patch_evaluation(self._recipe_cfg)  # for OTX compatibility
        if self._recipe_cfg.get("evaluation", None):
            self.metric = self._recipe_cfg.evaluation.metric

        if self._recipe_cfg.get("override_configs", None):
            self.override_configs.update(self._recipe_cfg.override_configs)

        if not self.freeze:
            remove_from_configs_by_type(self._recipe_cfg.custom_hooks, "FreezeLayers")

    def _update_stage_module(self, stage_module: str):
        module_prefix = {TrainType.Semisupervised: "SemiSL", TrainType.Incremental: "Incr"}
        if self._train_type == TrainType.Semisupervised and stage_module == "SegExporter":
            stage_module = "SemiSLSegExporter"
        elif self._train_type in module_prefix and stage_module in ["SegTrainer", "SegInferrer"]:
            stage_module = module_prefix[self._train_type] + stage_module

        return stage_module

    def _init_model_cfg(self):
        model_cfg = MPAConfig.fromfile(os.path.join(self._model_dir, "model.py"))
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

    def _add_predictions_to_dataset(self, prediction_results, dataset, dump_soft_prediction):
        """Loop over dataset again to assign predictions. Convert from MMSegmentation format to OTX format."""

        for dataset_item, (prediction, feature_vector) in zip(dataset, prediction_results):
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

            if dump_soft_prediction:
                for label_index, label in self._label_dictionary.items():
                    if label_index == 0:
                        continue
                    current_label_soft_prediction = soft_prediction[:, :, label_index]
                    class_act_map = get_activation_map(current_label_soft_prediction)
                    result_media = ResultMediaEntity(
                        name=label.name,
                        type="soft_prediction",
                        label=label,
                        annotation_scene=dataset_item.annotation_scene,
                        roi=dataset_item.roi,
                        numpy=class_act_map,
                    )
                    dataset_item.append_metadata_item(result_media, model=self._task_environment.model)

    def _initialize_post_hook(self, options=None):
        super()._initialize_post_hook(options)
        options["model_builder"] = build_segmentor
