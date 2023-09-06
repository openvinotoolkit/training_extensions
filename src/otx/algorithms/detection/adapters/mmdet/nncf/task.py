"""NNCF Task of OTX Detection."""

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


from functools import partial
from typing import Optional

import otx.algorithms.detection.adapters.mmdet.nncf.patches  # noqa: F401  # pylint: disable=unused-import
from otx.algorithms.common.tasks.nncf_task import NNCFBaseTask
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.nncf import build_nncf_detector
from otx.algorithms.detection.adapters.mmdet.task import MMDetectionTask
from otx.algorithms.detection.adapters.mmdet.utils.config_utils import (
    should_cluster_anchors,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import ModelEntity
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper

logger = get_logger()


# pylint: disable=too-many-ancestors
class DetectionNNCFTask(NNCFBaseTask, MMDetectionTask):
    """DetectionNNCFTask."""

    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__()
        super(NNCFBaseTask, self).__init__(task_environment, output_path)
        self._set_attributes_by_hyperparams()

    def _init_task(self, dataset: Optional[DatasetEntity] = None, export: bool = False):
        super(NNCFBaseTask, self)._init_task(dataset, export)
        self._prepare_optimize(export)

    def _prepare_optimize(self, export=False):
        super()._prepare_optimize()

        self.model_builder = partial(
            self.model_builder,
            nncf_model_builder=build_nncf_detector,
            return_compression_ctrl=False,
            is_export=export,
        )

    def _optimize(
        self,
        dataset: DatasetEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        results = self._train_model(dataset)

        return results

    def _optimize_post_hook(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
    ):
        # get prediction on validation set
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        val_preds, val_map = self._infer_model(val_dataset, InferenceParameters(is_evaluation=True))

        preds_val_dataset = val_dataset.with_empty_annotations()
        self._add_predictions_to_dataset(val_preds, preds_val_dataset, 0.0)

        result_set = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=val_dataset,
            prediction_dataset=preds_val_dataset,
        )

        # adjust confidence threshold
        if self._hyperparams.postprocessing.result_based_confidence_threshold:
            best_confidence_threshold = None
            logger.info("Adjusting the confidence threshold")
            metric = MetricsHelper.compute_f_measure(result_set, vary_confidence_threshold=True)
            if metric.best_confidence_threshold:
                best_confidence_threshold = metric.best_confidence_threshold.value
            if best_confidence_threshold is None:
                raise ValueError("Cannot compute metrics: Invalid confidence threshold!")
            logger.info(f"Setting confidence threshold to {best_confidence_threshold} based on results")
            self.confidence_threshold = best_confidence_threshold
        else:
            metric = MetricsHelper.compute_f_measure(result_set, vary_confidence_threshold=False)

        performance = metric.get_performance()
        logger.info(f"Final model performance: {str(performance)}")
        performance.dashboard_metrics.extend(
            # pylint: disable-next=protected-access
            self._generate_training_metrics(self._learning_curves, val_map)
        )
        output_model.performance = performance

    def _save_model_post_hook(self, modelinfo):
        if self._recipe_cfg is not None and should_cluster_anchors(self._recipe_cfg):
            modelinfo["anchors"] = {}
            self._update_anchors(modelinfo["anchors"], self.config.model.bbox_head.anchor_generator)

        modelinfo["confidence_threshold"] = self.confidence_threshold
