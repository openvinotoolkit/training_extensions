"""NNCF Task for OTX Classification."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import List, Optional

import otx.algorithms.classification.adapters.mmcls.nncf.patches  # noqa: F401  # pylint: disable=unused-import
import otx.algorithms.classification.adapters.mmcls.nncf.registers  # noqa: F401  # pylint: disable=unused-import
from otx.algorithms.classification.adapters.mmcls.nncf.builder import (
    build_nncf_classifier,
)
from otx.algorithms.classification.adapters.mmcls.task import MMClassificationTask
from otx.algorithms.common.tasks.nncf_task import NNCFBaseTask
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.metrics import (
    CurveMetric,
    LineChartInfo,
    LineMetricsGroup,
    MetricsGroup,
    Performance,
    ScoreMetric,
)
from otx.api.entities.model import ModelEntity  # ModelStatus
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.task_environment import TaskEnvironment
from otx.utils.logger import get_logger

logger = get_logger()


class ClassificationNNCFTask(NNCFBaseTask, MMClassificationTask):  # pylint: disable=too-many-ancestors
    """ClassificationNNCFTask."""

    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__()
        super(NNCFBaseTask, self).__init__(task_environment, output_path)
        self._set_attributes_by_hyperparams()

    def configure(
        self,
        training=True,
        ir_options=None,
        export=False,
    ):
        """Configure configs for nncf task."""
        super(NNCFBaseTask, self).configure(training, ir_options, export)
        self._prepare_optimize(export)
        return self._config

    def _prepare_optimize(self, export=False):
        super()._prepare_optimize()

        self.model_builder = partial(
            self.model_builder,
            nncf_model_builder=build_nncf_classifier,
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
        # Get training metrics group from learning curves
        training_metrics, final_acc = self._generate_training_metrics_group(self._learning_curves)
        performance = Performance(
            score=ScoreMetric(value=final_acc, name="accuracy"),
            dashboard_metrics=training_metrics,
        )

        logger.info(f"Final model performance: {str(performance)}")
        output_model.performance = performance

    def _generate_training_metrics_group(self, learning_curves):
        """Parses the classification logs to get metrics from the latest training run.

        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        if self._multilabel:
            metric_key = "val/accuracy-mlc"
        elif self._hierarchical:
            metric_key = "val/MHAcc"
        else:
            metric_key = "val/accuracy_top-1"

        # Learning curves
        best_acc = -1
        if learning_curves is None:
            return output

        for key, curve in learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            if key == metric_key:
                best_acc = max(curve.y)
            visualization_info = LineChartInfo(name=key, x_axis_label="Timestamp", y_axis_label=key)
            output.append(LineMetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        return output, best_acc

    def _save_model_post_hook(self, modelinfo):
        modelinfo["input_size"] = self._input_size
