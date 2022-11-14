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
from mpa import MPAConstants
from mpa.utils.config_utils import MPAConfig
from mpa.utils.logger import get_logger

from otx.algorithms.common.adapters.mmcv.utils import remove_from_config
from otx.algorithms.common.configs import TrainType
from otx.algorithms.common.tasks import BaseTask
from otx.algorithms.common.utils.callback import InferenceProgressCallback
from otx.algorithms.segmentation.adapters.mmseg.utils import (
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
from otx.api.utils.argument_checks import check_input_parameters_type
from otx.api.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
    create_hard_prediction_from_soft_prediction,
)

logger = get_logger()


# pylint: disable=too-many-locals
class SegmentationInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IUnload):
    """Inference Task Implementation of OTX Segmentation."""

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        # self._should_stop = False
        self.freeze = True
        self.metric = "mDice"
        self._label_dictionary = {}  # type: Dict
        super().__init__(SegmentationConfig, task_environment)

    def infer(
        self, dataset: DatasetEntity, inference_parameters: Optional[InferenceParameters] = None
    ) -> DatasetEntity:
        """Main infer function of OTX Segmentation."""
        logger.info("infer()")
        dump_features = True

        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            is_evaluation = inference_parameters.is_evaluation
        else:
            update_progress_callback = default_infer_progress_callback
            is_evaluation = False

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        stage_module = "SegInferrer"
        self._data_cfg = self._init_test_data_cfg(dataset)
        self._label_dictionary = dict(enumerate(self._labels, 1))
        results = self._run_task(stage_module, mode="train", dataset=dataset, dump_features=dump_features)
        logger.debug(f"result of run_task {stage_module} module = {results}")
        predictions = results["outputs"]
        prediction_results = zip(predictions["eval_predictions"], predictions["feature_vectors"])
        self._add_predictions_to_dataset(prediction_results, dataset, dump_soft_prediction=not is_evaluation)
        return dataset

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
        self._delete_scratch_space()

    def export(self, export_type: ExportType, output_model: ModelEntity):
        """Export function of OTX Detection Task."""
        logger.info("Exporting the model")
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f"not supported export type {export_type}")
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        stage_module = "SegExporter"
        results = self._run_task(stage_module, mode="train", precision="FP32", export=True)
        results = results.get("outputs")
        logger.debug(f"results of run_task = {results}")
        if results is None:
            logger.error("error while exporting model result is None")
            # output_model.model_status = ModelStatus.FAILED
        else:
            bin_file = results.get("bin")
            xml_file = results.get("xml")
            if xml_file is None or bin_file is None:
                raise RuntimeError("invalid status of exporting. bin and xml should not be None")
            with open(bin_file, "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            with open(xml_file, "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            output_model.precision = [ModelPrecision.FP32]
            output_model.optimization_methods = self._optimization_methods
            output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
        logger.info("Exporting completed")

    def _init_recipe_hparam(self) -> dict:
        warmup_iters = int(self._hyperparams.learning_parameters.learning_rate_warmup_iters)
        lr_config = (
            ConfigDict(warmup_iters=warmup_iters)
            if warmup_iters > 0
            else ConfigDict(warmup_iters=warmup_iters, warmup=None)
        )

        if self._hyperparams.learning_parameters.enable_early_stopping:
            early_stop = ConfigDict(
                start=int(self._hyperparams.learning_parameters.early_stop_start),
                patience=int(self._hyperparams.learning_parameters.early_stop_patience),
                iteration_patience=int(self._hyperparams.learning_parameters.early_stop_iteration_patience),
            )
        else:
            early_stop = False

        return ConfigDict(
            optimizer=ConfigDict(lr=self._hyperparams.learning_parameters.learning_rate),
            lr_config=lr_config,
            early_stop=early_stop,
            data=ConfigDict(
                samples_per_gpu=int(self._hyperparams.learning_parameters.batch_size),
                workers_per_gpu=int(self._hyperparams.learning_parameters.num_workers),
            ),
            runner=ConfigDict(max_epochs=int(self._hyperparams.learning_parameters.num_iters)),
        )

    def _init_recipe(self):
        logger.info("called _init_recipe()")

        recipe_root = os.path.join(MPAConstants.RECIPES_PATH, "stages/segmentation")
        train_type = self._hyperparams.algo_backend.train_type
        logger.info(f"train type = {train_type}")

        # load INCREMENTAL recipe file first. (default train type)
        recipe = os.path.join(recipe_root, "class_incr.py")
        
        if train_type != TrainType.INCREMENTAL:
            if train_type == TrainType.SEMISUPERVISED:
                if self._data_cfg.get('data', None) and self._data_cfg.data.get('unlabeled', None):
                    recipe = os.path.join(recipe_root, "cutmix_seg.py")
                else:
                    logger.warning(f"Cannot find unlabeled data.. convert to INCREMENTAL.")
                    train_type = TrainType.INCREMENTAL
            elif train_type == TrainType.SELFSUPERVISED:
                # recipe = os.path.join(recipe_root, 'pretrain.yaml')
                raise NotImplementedError(f"Train type {train_type} is not implemented yet.")
            else:
                # raise NotImplementedError(f'train type {train_type} is not implemented yet.')
                # FIXME: Temporary remedy for CVS-88098
                logger.warning(f"Train type {train_type} is not implemented yet.. convert to INCREMENTAL.")
                train_type = TrainType.INCREMENTAL

        logger.info(f"train type = {train_type} - loading {recipe}")

        self._recipe_cfg = MPAConfig.fromfile(recipe)
        patch_datasets(self._recipe_cfg)  # for OTX compatibility
        patch_evaluation(self._recipe_cfg)  # for OTX compatibility
        self.metric = self._recipe_cfg.evaluation.metric
        if not self.freeze:
            remove_from_config(self._recipe_cfg, "params_config")
        logger.info(f"initialized recipe = {recipe}")

    def _init_model_cfg(self):
        base_dir = os.path.abspath(os.path.dirname(self.template_file_path))
        return MPAConfig.fromfile(os.path.join(base_dir, "model.py"))

    def _init_test_data_cfg(self, dataset: DatasetEntity):
        data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    dataset=ConfigDict(
                        otx_dataset=None,
                        labels=self._labels,
                    )
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
                        name="Soft Prediction",
                        type="soft_prediction",
                        label=label,
                        annotation_scene=dataset_item.annotation_scene,
                        roi=dataset_item.roi,
                        numpy=class_act_map,
                    )
                    dataset_item.append_metadata_item(result_media, model=self._task_environment.model)
